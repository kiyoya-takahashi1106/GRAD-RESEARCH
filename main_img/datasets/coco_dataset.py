import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import sys
import json
from pathlib import Path

from PIL import Image


class COCODataset(Dataset):
    def __init__(self, split: str, root: str = "../data"):
        self.root = Path(root)
        if(split == 'all'):
            self.splits = ['train', 'val']
        else:
            self.splits = [split]

        self.samples = []
        for split in self.splits:
            self.get_img_path_and_text(split)


    def get_img_path_and_text(self, split: str):
        json_file = self.root / "coco" / "annotations" / f"captions_{split}2017.json"
        with open(json_file, "r") as f:
            data = json.load(f)
            
            for ann in data["annotations"]:
                image_id = ann["image_id"]
                caption = ann["caption"]
                img_filename = f"{image_id:012d}.jpg"
                img_path = self.root / "coco" / f"{split}2017" / img_filename

                # 実際のファイル存在チェック
                if not img_path.is_file():
                    print(f"[WARN] skip ({split}): {img_filename} -> {img_path} (not found)")
                    continue

                # print(split, caption, img_path)

                self.samples.append((split, caption, img_path))


    def __len__(self) -> int:
        return len(self.samples)


    def __getitem__(self, idx):
        split, caption, img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return split, caption, img