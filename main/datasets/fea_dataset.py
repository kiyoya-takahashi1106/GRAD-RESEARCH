"""
trainDataの保管しているRoBERTaとWav2Vec2のembeddingを返すDataset
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
    def __init__(self, dataset: str, split: str, hidden_dim: int, root: str = "../data"):
        self.dataset = dataset
        self.split = split

        if (dataset == "audiocaps"):
            if (split == 'all'):
                self.splits = ['train', 'val', 'test']
            elif (split == "train"):
                self.splits = ['train', 'val']   # trainの全部とvalの37個以外
            elif (split == "val"): 
                self.splits = ['val', 'test']   # valのうち前37個のみとtest全部
                
        elif (dataset == "fsd50k"):
            # No test split Dataset 
            if (split == 'all'):
                self.splits = ['train', 'val']
            elif (split == "train"):
                self.splits = ['train', 'val']   # valの1000個以外はtrainに含まれる
            elif (split == "val"):
                self.splits = ['val']            # valのうち前1000個のみ

        elif (dataset == "clotho"):
            if (split == 'all'):
                self.splits = ['train', 'val', 'test']
            elif (split == "train"):
                self.splits = ['train', 'val', 'test']   # trainとvalの全部とtestの1000個以外はtrain
            elif (split == "val"):
                self.splits = ['test']                   # testのうち前1000個のみ

        elif (dataset == "macs"):
            if (split == 'all'):
                self.splits = ['train']
            elif (split == "train"):
                self.splits = ['train']   # trainの1000個以外はtrain
            elif (split == "val"):
                self.splits = ['train']   # trainのうち前1000個のみ

        self.root = Path(root)

        self.samples = []
        for temp_split in self.splits:
            self.get_feature(dataset, temp_split, hidden_dim)


    def get_feature(self, dataset: str, temp_split: str, hidden_dim: int):
        # ❶ ここで一度だけ埋め込みファイルを読み込む
        fea_path = self.root / self.dataset / f"fea{hidden_dim}.pt"
        file_content = torch.load(fea_path, map_location="cpu")

        # ❷ サンプルをまとめて self.samples に展開
        features_lst = file_content[temp_split]

        if (dataset == "audiocaps"):
            if (self.split == "train" and temp_split == "val"):
                features_lst = features_lst[37:]   # valの37個以外はtrainに含まれる
            elif (self.split == "val" and temp_split == "val"):
                features_lst = features_lst[:37]   # valのうち前37個のみ

        if (dataset == "fsd50k"):
            if (self.split == "train" and temp_split == "val"):
                features_lst = features_lst[1000:]   # valの1000個以外はtrainに含まれる
            elif (self.split == "val" and temp_split == "val"):
                features_lst = features_lst[:1000]   # valのうち前1000個のみ

        elif (dataset == "clotho"):
            if (self.split == "train" and temp_split == "test"):
                features_lst = features_lst[1000:]   # testの1000個以外はtrainに含まれる
            elif (self.split == "val" and temp_split == "test"):
                features_lst = features_lst[:1000]   # testのうち前1000個のみ

        elif (dataset == "macs"):
            if (self.split == "train" and temp_split == "train"):
                features_lst = features_lst[1000:]   # trainの1000個以外はtrainに含まれる
            elif (self.split == "val" and temp_split == "train"):
                features_lst = features_lst[:1000]   # trainのうち前1000個のみ

        for text_emb, audio_emb in features_lst:
            self.samples.append((text_emb, audio_emb))


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]