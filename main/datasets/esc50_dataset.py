import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio

import os
import sys
import pandas as pd
from pathlib import Path


class ESC50Dataset(Dataset):
    def __init__(self, text_tokenizer, audio_processor, split: str, root: str = "../data"):
        self.text_tokenizer = text_tokenizer
        self.audio_processor = audio_processor

        self.metadata_path = Path(root) / "ESC-50-master" / "meta" / "esc50.csv"
        self.metadata = pd.read_csv(self.metadata_path)

        self.prompt = 'this is the sound of'
        self.classes = self.get_class()
        self.input_ids = []
        self.attn_masks = []
        self.text_preprocess()

        self.samples = []
        self.get_audio_path()


    # get class, add prompt
    def get_class(self):
        classes = []
        for line in self.metadata:
            category = line['category']
            if (category not in classes):
                text = self.prompt + category
                classes.append(text)
        return classes

    
    # process tokenize
    def text_preprocess(self):
        tokenized = self.text_tokenizer(
            self.classes,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.input_ids = tokenized.input_ids
        self.attn_masks = tokenized.attention_mask
    

    def get_audio_path(self):
        for _, row in self.metadata.iterrows():
            filename = row['filename']
            class_ = row['category']
            audio_path = Path("../data") / "ESC-50-master" / "audio" / f"{filename}.wav"
            self.samples.append((audio_path, class_))


    def __len__(self) -> int:   
        return len(self.samples)


    # only related audio
    def __getitem__(self, idx):
        audio_path, class_ = self.samples[idx]
        
        TARGET_SR = 16000
        target_len = 16000 * 5
        wav, sr = torchaudio.load(audio_path)
        if (wav.size(0) > 1):
            wav = wav.mean(dim=0, keepdim=True)
        if (sr != TARGET_SR):
            wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        wav = wav.squeeze(0)

        wav, attn = self.random_crop_or_pad(wav, target_len)

        # すでに長さは揃っているので padding=False
        processor_output = self.audio_processor(
            wav,        
            sampling_rate=16000,
            padding=False,
            return_tensors="pt",
        )
        audio_input_values = processor_output["input_values"]
        audio_input_values = audio_input_values.squeeze(0)

        # one-hot vector
        class_ = self.prompt + class_
        one_hot_vec = torch.zeros(len(self.classes), dtype=torch.float)
        class_index = self.classes.index(class_)
        one_hot_vec[class_index] = 1.0

        return audio_input_values, attn, one_hot_vec

    
    def random_crop_or_pad(self, wav: torch.Tensor, target_len: int = 16000 * 5):
        L = wav.size(0)

        if (L >= target_len):
            if (L > target_len):
                start = torch.randint(0, L - target_len + 1, (1,)).item()
                wav = wav[start:start + target_len]
            attn = torch.ones(target_len, dtype=torch.long)
            return wav, attn

        # 5秒より短い場合 → 右側ゼロパディング
        pad_len = target_len - L
        padded = torch.cat([wav, torch.zeros(pad_len, device=wav.device)], dim=0)

        attn = torch.cat([
            torch.ones(L, dtype=torch.long, device=wav.device),
            torch.zeros(pad_len, dtype=torch.long, device=wav.device)
        ], dim=0)

        return padded, attn