import os
from datasets import load_dataset

data = load_dataset(
    "sayakpaul/nyu_depth_v2",
    split="train[:100]",
    trust_remote_code=True   
)


