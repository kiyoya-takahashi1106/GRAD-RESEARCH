import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

import os
import sys
import yaml
from pathlib import Path


class MacsDataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'  or split == 'train'):
            self.splits = ['train']   # trainのみしかない
        else:
            self.splits = []

        self.samples = []
        for split in self.splits:
            self.get_audio_path_and_text(split)


    def get_audio_path_and_text(self, split: str):
        yaml_file = self.root / "macs" / f"MACS.yaml"
            
        with open(yaml_file, 'r') as f:
            datas = yaml.safe_load(f)
        datas = datas["files"]

        for data in datas:
            filename = data["filename"]
            audio_path = f"{self.root}/macs/audio/{filename}"
            audio_path = Path(audio_path)

            annotations = data["annotations"]
            for annotation in annotations:
                caption = annotation["sentence"]
                # 実際のファイル存在チェック
                if (not audio_path.is_file()):
                    print(f"[WARN] skip ({split}) -> {audio_path} (not found)")
                    continue

                self.samples.append((split, caption, str(audio_path)))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx):
        split, caption, path = self.samples[idx]
        return split, caption, path