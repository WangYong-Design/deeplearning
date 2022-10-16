import torch
from torch.utils.data import Dataset,DataLoader
import os

class SimpleDataset(Dataset):
    def __init__(self,x,y) -> None:
        super().__init__()
        self.X = x
        self.Y = y
    
    def __len__(self):
        "Return data counts"
        return len(self.X)

    def __getitem__(self, index):
        "Get dataset,comapny with dataloader"
        return self.X[index],self.Y[index]






