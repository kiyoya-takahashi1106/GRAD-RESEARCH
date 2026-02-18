import torch
import torch.nn as nn
from torch.utils.data import Dataset

import os
import sys
import pandas as pd
from PIL import Image
from pathlib import Path


class Caltech101Dataset(Dataset):
    def __init__(self, text_tokenizer, img_processor, split: str, root: str = "../data"):
        self.text_tokenizer = text_tokenizer
        self.img_processor = img_processor

        self.root = root

        self.prompt = 'this is the photo of a'
        self.classes = self.get_class()
        self.input_ids = []
        self.attn_masks = []
        self.text_preprocess()

        self.samples = self.get_img_path()


    # add prompt
    def get_class(self):
        classes = []
        root_dir = os.path.join(self.root, "caltech101/101_ObjectCategories")
        for name in os.listdir(root_dir):
            text = f"{self.prompt} {name}"
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
    

    def get_img_path(self):
        samples = []
        img_folder = os.path.join(self.root, "caltech101/101_ObjectCategories")
        for class_ in os.listdir(img_folder):
            for img_file in os.listdir(os.path.join(img_folder, class_)):
                prompt_class_ = f"{self.prompt} {class_}"
                samples.append((f"{img_folder}/{class_}/{img_file}", prompt_class_))
        return samples


    def __len__(self) -> int:   
        return len(self.samples)


    # only related image
    def __getitem__(self, idx):
        img_path, class_ = self.samples[idx]
        img_path = Path(img_path)
        img = Image.open(img_path).convert("RGB")
        # print("loading:", img_path)

        # すでに長さは揃っているので padding=False
        processor_output = self.img_processor(
            images=img,        
            padding=False,
            return_tensors="pt",
        )
        img_input_values = processor_output["pixel_values"]

        # one-hot vector
        one_hot_vec = torch.zeros(len(self.classes), dtype=torch.float)
        class_index = self.classes.index(class_)
        one_hot_vec[class_index] = 1.0

        return img_input_values, None, one_hot_vec