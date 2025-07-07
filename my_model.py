import torch
import torch.nn as nn 
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
from einops import rearrange

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
    def __init__(self, ):
        super().__init__()

        # Define number of Mask and mode 

        # Patch Embeddings and Pos Embed

        # adding CLS

        # Layer Norm


        # Teacher and Student Encoder, where student encoder is the deepcopy of the teacher encoder


        # Initialize Predictor

    @torch.no_grad()
    def get_target(self):
        
        pass

    def get_context(self):

        pass

    def forward(self):

        pass

    
        
    

    

    
