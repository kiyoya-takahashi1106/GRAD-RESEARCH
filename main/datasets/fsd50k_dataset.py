import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

import os
import sys
import json
from pathlib import Path


class FSD50KDataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'):
            self.splits = ['train', 'val']
        else:
            self.splits = [split]

        self.samples = []
        for split in self.splits:
            self.get_audio_path_and_text(split)


    def get_audio_path_and_text(self, split: str):
        if (split == 'train'):
            json_file = self.root / "fsd50k" / "FSD50K.metadata"  / "dev_clips_info_FSD50K.json"
        elif (split == 'val'):
            json_file = self.root / "fsd50k" / "FSD50K.metadata"  / "eval_clips_info_FSD50K.json"
            
        with open(json_file, 'r') as f:
            data = json.load(f)

        for data_id, info in data.items():
            if (split == 'train'):
                audio_path = f"{self.root}/fsd50k/FSD50K.dev_audio_16k/{data_id}.wav"
            elif (split == 'val'):
                audio_path = f"{self.root}/fsd50k/FSD50K.eval_audio_16k/{data_id}.wav"

            title = info["title"]
            description = info["description"]
            caption = f"{title}. {description}"
            
            # 実際のファイル存在チェック
            audio_path = Path(audio_path)
            if (not audio_path.is_file()):
                print(f"[WARN] skip ({split}): {data_id} -> {audio_path} (not found)")
                continue

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