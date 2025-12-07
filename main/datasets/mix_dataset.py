import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path

from datasets.audiocaps_dataset import AudioCapsDataset
from datasets.fsd50k_dataset import FSD50KDataset


class MixDataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'):
            self.splits = ['train', 'val', 'test']
        else:
            self.splits = [split]

        self.audiocaps_dataset = AudioCapsDataset(split, root)
        self.fsd50k_dataset = FSD50KDataset(split, root)

        self.samples = []
        for dataset in [self.audiocaps_dataset, self.fsd50k_dataset]:
            for i in range(len(dataset)):
                self.samples.append(dataset[i])


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, i: int):
        return self.samples[i]