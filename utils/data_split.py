from torch.utils.data import random_split
import torch

def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, seed=42):
    length = len(dataset)
    train_len = int(length * train_ratio)
    val_len = int(length * val_ratio)
    test_len = length - train_len - val_len
    assert train_len + val_len + test_len == length

    generator = torch.Generator().manual_seed(seed)
    
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)