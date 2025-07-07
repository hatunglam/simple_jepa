from my_model import Predictor
import torch

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

if __name__ == "__main__":
    test_predictor()