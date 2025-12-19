import torch
import torch.nn as nn
import torch.nn.functional as F
print("CUDA available:", torch.cuda.is_available())

from msclap import CLAP
from tqdm import tqdm

import os
import sys
import numpy as np
import argparse
from functools import partial

from datasets.audiocaps_dataset import AudioCapsDataset
from datasets.fsd50k_dataset import FSD50KDataset
from datasets.clotho_dataset import ClothoDataset
from datasets.macs_dataset import MacsDataset
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.utility import set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, help="audiocaps, fsd50k, clotho, macs")
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

set_seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
clap_model = CLAP(version = '2022', use_cuda=True)

# Load dataset
if (args.dataset == "audiocaps"):
    dataset = AudioCapsDataset(split="all")
elif (args.dataset == "fsd50k"):
    dataset = FSD50KDataset(split="all")
elif (args.dataset == "clotho"):
    dataset = ClothoDataset(split="all")
elif (args.dataset == "macs"):
    dataset = MacsDataset(split="all")
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

print(f"Dataset: {args.dataset}, #samples: {len(dataset)}")

"""
{
    train: [(text_fea, audio_fea), ...],
      val: [(text_fea, audio_fea), ...],
     test: [(text_fea, audio_fea), ...],
}
"""

fea_dct = {
    'train': [],
    'val': [],
    'test': []
}

# Computing fea
with torch.no_grad():   
    train_bar = tqdm(dataloader, leave=False)
    for batch in train_bar:
        splits, text_x, audio_path = batch

        # for i in range(len(text_x)):
        #     print(len(text_x[i]), audio_path[i])

        text_embedding = clap_model.get_text_embeddings(text_x)
        audio_embedding = clap_model.get_audio_embeddings(audio_path, resample=True)

        # save embedding
        for i in range(len(batch[0])):
            split = splits[i]
            text_embedding_i = text_embedding[i].unsqueeze(0).cpu()
            audio_embedding_i = audio_embedding[i].unsqueeze(0).cpu()
            fea_dct[split].append(
                (text_embedding_i, audio_embedding_i)
            )

# Save fea
save_path = f"../data/{args.dataset}/clap_fea.pt"
torch.save(fea_dct, save_path)
print(f"Saved fea to {save_path}")