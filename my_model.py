import torch
import torch.nn as nn 
import torch.nn.functional as F
from x_transformers import Encoder, Decoder
from einops import rearrange

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_size):
        super().__init__()
        if isinstance(img_size, int):
            img_size = img_size, img_size
        if isinstance(patch_size, int):
            patch_size = patch_size, patch_size

        # Calculate the number of patches
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.n_patches = (img_size // patch_size)**2

        self.conv = nn.Conv2d(in_channels, embed_size,
                              kernel_size= patch_size, stride = patch_size)
        
    def forward(self, x):
        x = self.conv(x)
        # reshape
        x = rearrange(x, "b e h w -> b (h w) e")
        
        return x
    

    
