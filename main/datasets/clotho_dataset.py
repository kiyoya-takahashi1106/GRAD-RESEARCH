import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

import os
import sys
import csv
from pathlib import Path


class ClothoDataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'):
            self.splits = ['train', 'val', 'test']
        else:
            self.splits = [split]

        self.samples = []
        for split in self.splits:
            self.get_audio_path_and_text(split)


    def get_audio_path_and_text(self, split: str):
        if (split == "train"):
            temp_split = "development"
        elif (split == "val"):
            temp_split = "validation"
        else:
            temp_split = "evaluation"

        csv_file = self.root / "clotho" / f"clotho_captions_{temp_split}.csv"
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                audio_filename = row[0]
                caption1 = row[1]
                caption2 = row[2]
                caption3 = row[3]
                caption4 = row[4]
                caption5 = row[5]

                audio_path = self.root / "clotho" / temp_split / Path(audio_filename)

                # 実際のファイル存在チェック
                if not audio_path.is_file():
                    print(f"[WARN] skip ({split}): {audio_filename} -> {audio_path} (not found)")
                    continue
            
                for caption in [caption1, caption2, caption3, caption4, caption5]:
                    self.samples.append((split, caption, audio_path))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx):
        split, caption, path = self.samples[idx]

        TARGET_SR = 16000
        wav, sr = torchaudio.load(path)
        if (wav.size(0) > 1):
            wav = wav.mean(dim=0, keepdim=True)
        if (sr != TARGET_SR):
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0)

        return split, caption, wav
