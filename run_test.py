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
    #print('2d id:', start_i, start_j)

    matrix = torch.arange(0, patch_h*patch_w).reshape(patch_h, patch_w)
    print(matrix)

    id_1d = (start_i * patch_w) + start_j
    #print(id_1d)

    return id_1d # (int) a single id for starting patch 

def get_target_block(patch_dim = (5, 8), scale = 0.2, aspect_ratio = 2, n_target_blocks=3):
    
    n_patch_h, n_patch_w = patch_dim

    # scale is the ratio of the target block to the whole image
    n_patches_perblock = int(n_patch_h * n_patch_w * scale)

    # print(patch_dim)
    # print(f'total patches: {n_patch_h*n_patch_w}')
    # print(n_patches_perblock)

    # Calculate the height and width and maintain aspect ratio
    # aspect ratio = h / w
    # <=> h = aspect ratio * w 
    # w * h <=> w^2 * aspect ratio = n_patches_perblock
    # <=> w = sqrt(n_patches_perblock / aspect ratio)
    
    block_w = int(torch.sqrt(torch.tensor(n_patches_perblock / aspect_ratio)))
    block_h = int(block_w * aspect_ratio)

    block_dim = block_h, block_w

    #print('Image shape (patches): ', n_patch_h, n_patch_w)
    #print('Block shape (patches): ', block_h, block_w)

    # Initialize placeholder
    target_patches = [] # [ [id11, id12, ...], [id21, id22, ...], ... ]
    all_patches = set() # membership check
    
    # Loop through all target blocks
    for target_id in range(n_target_blocks):
        start_patch = select_patches(patch_dim, block_dim)
        # --> id 
        #print("start patch id = ", start_patch)

        # List to hold the patches:
        patches = []

        # Collect patches  
        for h in range(block_h):
            for w in range(block_w):
                patch_i = start_patch + (h * n_patch_w) + w
                # start_patch + (h * n_patch_w) --> skip to next h row, same column
                # + w ---> move to the next w column 

                patches.append(patch_i) # append the patch id
                all_patches.add(patch_i)
                #print(f'pos {(h,w)}: {patch_i}')
        
        print(f"block {target_id}: patches = {patches}")

        # Finish 1 target block
        target_patches.append(patches) # append the whole block
        
        print("________________________________________________________________________________")
    
    return target_patches, all_patches


def generate_patch_id(nh= 5, nw= 8):
    h = torch.arange(nh)
    w = torch.arange(nw)
    print("h: ", h)
    print("w: ", w)

    grid = torch.cartesian_prod(h,w)

    print("grid: ", grid)

def generate_context(patch_dim, target_patches, scale= 0.2, aspect_ratio=2): # --> [id, id, ...]
    print('____________________________________________________________')
    patch_h, patch_w = patch_dim

    n_patches_perblock = int(patch_h * patch_w * scale)

    # Calculate height and width 
    block_w = int(torch.sqrt(torch.tensor(n_patches_perblock / aspect_ratio)))
    block_h = int(block_w * aspect_ratio)

    block_dim = block_h, block_w
    print(block_dim)

    start_patch = select_patches(patch_dim, block_dim)
    context_set = set()

    for h in range(block_h):
        for w in range(block_w):
            patch_hw = start_patch + (h * patch_w) + w

            context_set.add(patch_hw)

    print("Initial Context patches: ", context_set),
    print("target patches: ", target_patches)

    # Rule out match patches
    context_patches = list(context_set.difference(target_patches))
    print("final context: ", context_patches)

    return context_patches

def make_pred(n_target_blocks, batch,
               n_patches, # number of patches in a single target block
               embed_dim,
               target_patches, # [ [id, id, ...], [id, id, ...] ]
               context_encoding 
               ):
    
    n_image_patches = 8 * 5
    pos_embedding = torch.randn(1, n_image_patches, embed_dim)
    # n_patches: number of patches in an image
    # this is the positional embedding for all patches in the image
    print("pos_embedding shape: " ,  pos_embedding.shape)

    pos_ = torch.arange(n_image_patches).unsqueeze(1).repeat(1, embed_dim)
    pos = pos_.unsqueeze(0)
    print(pos)

    print("____________________________________________________________")
    mask_token = torch.randn(1, 1, embed_dim)
    print('Mask token shape: ', mask_token.shape)
    # a single vector with size of embedding

    # Initialize tensor to hold prediction block
    prediction_block = torch.zeros(n_target_blocks, batch, n_patches, embed_dim)
    print("____________________________________________________________")

    # Predict each target block seperately
    for block_id in range(n_target_blocks):
        
        target_mask = mask_token.repeat(batch, n_patches, 1)
        # clone batch times, n_patches times that single vector size embed_dim

        #print(target_mask.shape)
        print(f"block {block_id}")

        print("target_patches[block_id]: ", target_patches[block_id])
        print("queried: ", pos[:, target_patches[block_id], :])

        target_pos_emb = pos_embedding[:, target_patches[block_id], :]

def extract_context(x, context):
    # Context --> [id, id, ....] 
    return x[:, context, :]

def main_1():
    img_patch_dim = (5, 8)
    emb = 10
    scale = 0.2
    aspect_ratio = 2
    n_target_blocks=3

    h = 5
    w = 8 

    x = torch.arange(0, h*w).unsqueeze(1).repeat(1, emb).unsqueeze(0)
    print(x.shape)
    print(x)

    target, target_patches = get_target_block()
    # # # print("target shape: ", len(target))
    # # # make_pred(3, 5, 6, 10, target, None)
    context = generate_context(img_patch_dim, target_patches, scale= 0.2, aspect_ratio=2)

    print("###################################################################################")

    z = extract_context(x, context=context)
    print(z)

if __name__ == "__main__":
    pass

