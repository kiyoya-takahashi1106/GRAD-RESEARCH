"""
dataのpathとテキストそのものではなく、保管しているCLAPのembeddingを返すDataset
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


class AudioCapsFeaDataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'):
            self.splits = ['train', 'val', 'test']
        else:
            self.splits = [split]

        self.samples = []
        for split in self.splits:
            self.get_feature(split)


    def get_feature(self, split: str):
        # ❶ ここで一度だけ埋め込みファイルを読み込む
        fea_path = self.root / "audiocaps" / "clap_embeddings.pt"
        file_content = torch.load(fea_path, map_location="cpu")

        # ❷ サンプルをまとめて self.samples に展開
        self.samples = []
        features_lst = file_content[split]
        for text_emb, audio_emb in features_lst:
            # 必要ならラベル情報をここに足す
            self.samples.append((text_emb, audio_emb))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, i: int):
        return self.samples[i]