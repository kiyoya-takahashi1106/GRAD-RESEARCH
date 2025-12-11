"""
保管しているRoBERTaとWav2Vec2のembeddingを返すDataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path


"""
pt.fileの中身のイメージ
{
    train: [(text_embeddings, audio_embeddings), ...],
      val: [(text_embeddings, audio_embeddings), ...],
     test: [(text_embeddings, audio_embeddings), ...],
}
"""


class FeaDataset(Dataset):
    def __init__(self, dataset: str, split: str, root: str = "../data"):
        self.dataset = dataset
        if (dataset == "audiocaps"):
            if(split == 'all'):
                self.splits = ['train', 'val', 'test']
            else:
                self.splits = [split]
        elif (dataset == "fsd50k"):
            if(split == 'all'):
                self.splits = ['train', 'val']
            else:
                self.splits = [split]
        self.root = Path(root)

        self.samples = []
        for split in self.splits:
            self.get_feature(split)


    def get_feature(self, split: str):
        # ❶ ここで一度だけ埋め込みファイルを読み込む
        fea_path = self.root / self.dataset / "fea.pt"
        file_content = torch.load(fea_path, map_location="cpu")

        # ❷ サンプルをまとめて self.samples に展開
        self.samples = []
        features_lst = file_content[split]
        for text_emb, audio_emb in features_lst:
            self.samples.append((text_emb, audio_emb))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]