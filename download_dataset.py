import os
from datasets import load_dataset

data = load_dataset(
    "sayakpaul/nyu_depth_v2",
    split="train[:500]",
    cache_dir="./dataset"
)


