from my_model import Predictor
from model import IJEPA_base
import torch
from draw_tensor import draw

def test_predictor():
    embed_dim = 64
    batch_size = 3
    depth = 5
    n_heads = 8

    target_len = 3
    context_len = 2

    context = torch.randn(batch_size, context_len, embed_dim)
    target  = torch.randn(batch_size, target_len, embed_dim)

    model = Predictor(embed_dim, depth, n_heads)

    out = model(context, target)

    print('Output shape', out.shape)

    assert out.shape == (batch_size, target_len, embed_dim), 'Wrong shape'

def  generate_img(n_images=3):
    img = torch.randn(1, 3, 10, 10)
    img_rand = img.repeat(n_images, 2, 1, 1)      # Simulating a batch of 3 images

    return  img_rand

def check_mask_token(embed_dim=64, total_patches=25):
    mask_token = torch.randn(1, 1, embed_dim)
    print('Mask token shape:', mask_token.shape)
    draw(mask_token.shape[0], mask_token.shape[1], mask_token.shape[2])    
    print('_________________________________')

    pos_embed = torch.randn(1, total_patches, embed_dim)
    print('Positional embedding shape:', pos_embed.shape)
    draw(pos_embed.shape[0], pos_embed.shape[1], pos_embed.shape[2])
    print('_________________________________')

    batch = 3
    n_patches_per_block = 5

    target_masks = mask_token.repeat(batch, n_patches_per_block, 1)
    draw(target_masks.shape[0], target_masks.shape[1], target_masks.shape[2])
    print('Target masks shape:', target_masks.shape)
    print('_________________________________')

    #target_pos = pos_embed[:, ]

def random_int(limit):
    'Generate a random integer'
    return torch.randint(0, limit, (1,)).item()

def select_patches(patch_dim, block_dim):

    patch_h, patch_w = patch_dim

    block_h, block_w = block_dim

    max_i = patch_h - block_h + 1
    max_j = patch_w - block_w + 1

    start_i = random_int(max_i)
    start_j = random_int(max_j)
    print(start_i, start_j)

    matrix = torch.arange(0, patch_h*patch_w).reshape(patch_h, patch_w)
    print(matrix)

    id_1d = (start_i * patch_w) + start_j
    print(id_1d)
    

if __name__ == "__main__":
    select_patches(
        patch_dim= (7, 8),
        block_dim= (3, 4)
    )