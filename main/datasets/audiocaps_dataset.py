import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

from pathlib import Path


class AudioCapsDataset(Dataset):
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
        tsv_file = self.root / "audiocaps" / f"audiocaps_{split}.tsv"
        with open(tsv_file, 'r') as f:
            for line in f:
                _, audio_filename, caption, _ = line.strip().split('\t')
                audio_path = self.root / Path(audio_filename)

                # 実際のファイル存在チェック
                if not audio_path.is_file():
                    print(f"[WARN] skip ({split}): {audio_filename} -> {audio_path} (not found)")
                    continue

                TARGET_SR = 16000
                wav, sr = torchaudio.load(audio_path)   # (C, T)
                if (wav.size(0) > 1):
                    wav = wav.mean(dim=0, keepdim=True)
                if (sr != TARGET_SR):
                    wav = torchaudio.functional.resample(wav, sr, TARGET_SR)

                self.samples.append((caption, wav))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, i: int):
        return self.samples[i]