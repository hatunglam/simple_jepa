import torch
import torch.nn as nn 
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
from einops import rearrange
import copy

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super().__init__()

        # probably not needed
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size

        # Calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])    

        self.conv = nn.Conv2d(in_channels, embed_size,
                              kernel_size= patch_size, stride = patch_size)
        
    def forward(self, x):
        x = self.conv(x)
        # reshape
        x = rearrange(x, "b e h w -> b (h w) e")
        
        return x
    
class Predictor(nn.Module):
    def __init__(self, embed_dim, depth, n_heads):
        super().__init__()

        # The Predictor is a Decoder block
        # depth is the number of Transformer blocks
        self.predictor = Decoder(dim = embed_dim, depth = depth, heads = n_heads)

    def forward(self, context_rep, target_mask):
        # Concatenate context and target representation along the sequence length 
        x = torch.cat((context_rep, target_mask), dim=1)
        # predict with the decoder
        self.predictor(x) # ---> (Batch, Context + Target sequence, Embed)
        # Choose only the target prediction
        # Which is (Batch, Target sequence, Embed)
        return x[:, -target_mask.shape[1]:, :]
    
# Main model
class IJEPA(nn.Module):
    def __init__(
            self,
            img_size, patch_size, in_channels, embed_size,
            n_heads, encoder_depth, predictor_depth,
            M =4, post_embed_norm = False,
            mode = 'train', layer_dropout = 0.
            ):
        super().__init__()

        # Define number of Mask and mode 
        self.M = M
        self.mode = mode
        self.layer_dropout = layer_dropout

        # Patch Embeddings and Pos Embed
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels,
                                      embed_size)
        self.patch_dim = (self.patch_embed.patch_shape[0], self.patch_embed.patch_shape[1])
        self.n_tokens = self.patch_embed.patch_shape[0] * self.patch_embed.patch_shape[1]
        self.pos_embedding = nn.Parameter(torch.randn(2, self.n_tokens, embed_size))
        
        # adding CLS and masked token
        self.masked_token = nn.Parameter(torch.randn(1, 1, embed_size))
        nn.init.trunc_normal_(self.masked_token, 0.02)

        # Layer Norm
        self.norm = nn.LayerNorm(embed_size)
        self.post_embed_norm = nn.LayerNorm if post_embed_norm else nn.Identity()

        # Teacher and Student Encoder, where student encoder is the deepcopy of the teacher encoder
        self.teacher_encoder = Encoder(
            dim= embed_size,
            heads = n_heads,
            depth = encoder_depth,
            layer_dropout = self.layer_dropout
        )

        self.student_encoder = copy.deepcopy(self.teacher_encoder).cuda()

        # Initialize Predictor
        self.predictor = Predictor(embed_size, predictor_depth, n_heads)

    @torch.no_grad()
    def get_target(self):
        
        pass

    def get_context(self):

        pass

    def forward(
            self, x, 
            target_aspect_ratio, target_scale,
            context_aspect_ratio, context_scale
            ):
        
        # get patch embeddings (ViT)
        x = self.patch_embed(x)
        b, n, e = x.shape

        # add positional embeddings (ViT)
        x = x + self.pos_embedding

        # Normalize (ViT)
        x = self.post_embed_norm(x)        
        
        if self.mode == 'test':
            return self.student_encoder(x)
        
        # get target embeddings
        target_block, target_patch, all_patches = get_target()


        # get context embeddings


        # get prediction block and predict iteratively (part of Prediction module)

        


        
    

    

    
