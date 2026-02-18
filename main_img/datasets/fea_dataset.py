"""
trainDataの保管しているRoBERTaとVITのembeddingを返すDataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from pathlib import Path


"""
pt.fileの中身のイメージ
{
    train: [(text_embeddings, image_embeddings), ...],
      val: [(text_embeddings, image_embeddings), ...],
}
"""


class FeaDataset(Dataset):
    def __init__(self, dataset: str, split: str, hidden_dim: int, root: str = "../data"):
        self.dataset = dataset
        self.split = split

        self.root = Path(root)

        self.samples = []
        self.get_feature(split, hidden_dim)


    def get_feature(self, split: str, hidden_dim: int):
        # ❶ ここで一度だけ埋め込みファイルを読み込む
        fea_path = self.root / self.dataset / f"fea{hidden_dim}_imageNet.pt"
        file_content = torch.load(fea_path, map_location="cpu")

        # ❷ サンプルをまとめて self.samples に展開
        features_lst = file_content[split]
        for text_emb, image_emb in features_lst:
            self.samples.append((text_emb, image_emb))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]