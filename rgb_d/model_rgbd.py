import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Encoder, Decoder
import copy

'''
PatchEmbed class, adapted from https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632 I think, but I dont have medium premium so idk
- This class is used to convert the image into patches using a convolutional layer
'''
class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size 
        #calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        #convolutional layer to convert the image into patches
        self.conv = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        

    def forward(self, x):
        x = self.conv(x)
        #flatten the patches
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x

'''Lightweight Predictor Module using VIT to predict target patches from context patches'''
class Predictor(nn.Module):
    def __init__(self, embed_dim, num_heads, depth):
        super().__init__()
        
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = num_heads)
    def forward(self, context_encoding, target_masks):
        x = torch.cat((context_encoding, target_masks), dim = 1)
        x = self.predictor(x)
        #return last len(target_masks) tokens
        l = x.shape[1]
        return x[:, l - target_masks.shape[1]:, :]
    
'''Main Model Class'''
class IDJEPA_base(nn.Module):
    def __init__(self, img_size, patch_size, in_chans_rgb, in_chans_depth , embed_dim, enc_depth,
                pred_depth, num_heads, post_emb_norm=False, M = 4, mode="train", layer_dropout=0.):
        super().__init__()
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #define the patch embedding and positional embedding
        self.patch_embed_rgb = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans_rgb, embed_dim=embed_dim)
        self.patch_embed_depth = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans_depth, embed_dim=embed_dim)

        self.patch_dim  = (self.patch_embed_rgb.patch_shape[0], self.patch_embed_rgb.patch_shape[1]) # = n_patches (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_tokens = self.patch_embed_rgb.patch_shape[0] * self.patch_embed_rgb.patch_shape[1] # = n_patches height * n_patches width = total n_patches
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim)) # initialize learnable parameter (seq_len, embed_dim)

        #define the mask tokens
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim)) # adding another token on top 
        nn.init.trunc_normal_(self.mask_token, 0.02)

        #define the encoder and decoder, as well as the layer normalization and dropout
        self.post_emb_norm = nn.LayerNorm(embed_dim) if post_emb_norm else nn.Identity()
        self.norm = nn.LayerNorm(embed_dim)
        self.teacher_encoder = Encoder(
            dim=embed_dim,
            heads=num_heads,
            depth=enc_depth, 
            layer_dropout=self.layer_dropout,
        )  

        # copy.deepcopy(self.teacher_encoder).cuda() if gpu is available
        self.student_encoder = copy.deepcopy(self.teacher_encoder)
        self.predictor = Predictor(embed_dim, num_heads, pred_depth)

    @torch.no_grad() 
    def get_target_block(self, target_encoder, x, patch_dim, aspect_ratio, scale, M):  
        #get the target block
        target_encoder = target_encoder.eval()
        x = target_encoder(x)  
        x = self.norm(x)
        #get the patch dimensions
        patch_h, patch_w = patch_dim # = n_patches for each side
        #get the number of patches
        num_patches = patch_h * patch_w
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale) # scale : % of the image used for each target block 
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the patches in the target block
        target_block = torch.zeros((M, x.shape[0], block_h*block_w, x.shape[2]))
        target_patches = []
        all_patches = []
        for z in range(M):
            #get the starting patch
            start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
            start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
            start_patch = start_patch_h * patch_w + start_patch_w

            patches = []
            #get the patches in the target block
            for i in range(block_h):
                for j in range(block_w):
                    patches.append(start_patch + i * patch_w + j)
                    if start_patch + i * patch_w + j not in all_patches:
                        all_patches.append(start_patch + i * patch_w + j)
                    
            #get the target block
            target_patches.append(patches)
            target_block[z] = x[:, patches, :]
        return target_block.to(self.device), target_patches, all_patches

    def get_context_block(self, x, patch_dim, aspect_ratio, scale, target_patches):
        patch_h, patch_w = patch_dim
        #get the number of patches in the target block
        num_patches_block = int(patch_h * patch_w * scale)
        #get the height and width of the target block with aspect ratio
        block_h = int(torch.sqrt(torch.tensor(num_patches_block / aspect_ratio)))
        block_w = int(aspect_ratio * block_h)
        #get the starting patch
        start_patch_h = torch.randint(0, patch_h - block_h+1, (1,)).item()
        start_patch_w = torch.randint(0, patch_w - block_w+1, (1,)).item()
        start_patch = start_patch_h * patch_w + start_patch_w
        #get the patches in the context_block
        patches = []
        for i in range(block_h):
            for j in range(block_w):
                if start_patch + i * patch_w + j not in target_patches: #remove the target patches
                    patches.append(start_patch + i * patch_w + j)
        return x[:, patches, :]


    def forward(self,data, target_aspect_ratio=1,
                target_scale=1, context_aspect_ratio=1,
                context_scale=1):
        
        # try having Input as Dict 
        # x: {"rgb": ..., "depth": ...}
        x_rgb, x_dep = data["rgb_image"], data["depth_image"] 
        # Check input shape
        assert x_rgb.shape[1] == 3, f"Expected RGB input with 3 channels, got {x_rgb.shape[1]}"
        assert x_dep.shape[1] == 1, f"Expected depth input with 1 channel, got {x_dep.shape[1]}"

        # --------rgb patch embeddings--------
        #get the patch embeddings
        x_rgb = self.patch_embed_rgb(x_rgb)
        b, n, e = x_rgb.shape
        #add the positional embeddings
        x_rgb = x_rgb + self.pos_embedding
        #normalize the embeddings
        x_rgb = self.post_emb_norm(x_rgb)
        #if mode is test, we get return full embedding:
        if self.mode == 'test':
            return self.student_encoder(x_rgb)

        # --------depth patch embeddings--------
        #get the patch embeddings
        x_dep = self.patch_embed_depth(x_dep)
        b, n, e = x_dep.shape
        #add the positional embeddings
        x_dep = x_dep + self.pos_embedding
        #normalize the embeddings
        x_dep = self.post_emb_norm(x_dep)

        #---------------------------------------

        # #get target embeddings
        target_blocks, target_patches, all_patches = self.get_target_block(
            self.teacher_encoder, x_dep, self.patch_dim,
            target_aspect_ratio, target_scale, self.M
            )
        
        # n = n_patches in each target block
        m, b, n, e = target_blocks.shape
        #get context embedding

        context_block = self.get_context_block(
            x_rgb, self.patch_dim, context_aspect_ratio, context_scale, all_patches
            )
        
        context_encoding = self.student_encoder(context_block)
        context_encoding = self.norm(context_encoding)


        prediction_blocks = torch.zeros((m, b, n, e)) # no .cuda()
        #get the prediction blocks, predict each target block separately
        for i in range(m):
            target_masks = self.mask_token.repeat(b, n, 1)
            target_pos_embedding = self.pos_embedding[:, target_patches[i], :] 
            # access the ith vector in list of patches, contains list of patch ids
            target_masks = target_masks + target_pos_embedding
            prediction_blocks[i] = self.predictor(context_encoding, target_masks)

        return prediction_blocks, target_blocks
    
    