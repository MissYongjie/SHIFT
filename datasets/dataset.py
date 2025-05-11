# datasets/dataset.py

import torch
from torch.utils.data import Dataset

class ChangeDetectionDataset(Dataset):
    def __init__(self, img_t1, img_t2, scribble):
        self.img_t1 = torch.from_numpy(img_t1.transpose(2, 0, 1)).float()
        self.img_t2 = torch.from_numpy(img_t2.transpose(2, 0, 1)).float()
        self.scribble = torch.from_numpy(scribble).float().unsqueeze(0)

    def __len__(self):
        return 1  # 当前数据集是单样本，后续可以扩展支持多样本

    def __getitem__(self, idx):
        return self.img_t1, self.img_t2, self.scribble

    def update_scribble(self, new_scribble):
        self.scribble = torch.from_numpy(new_scribble).float().unsqueeze(0)
