import torch
import torch.nn as nn
import torch.nn.functional as F
print("CUDA available:", torch.cuda.is_available())

from msclap import CLAP
from tqdm import tqdm

import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from datasets.audiocaps_dataset import AudioCapsDataset

# Load CLAP model
clap_model = CLAP(version='2023', use_cuda=torch.cuda.is_available())

# Load dataset
dataset = AudioCapsDataset(split="all")

"""
{
    train: [(text_embeddings, audio_embeddings), ...],
      val: [(text_embeddings, audio_embeddings), ...],
     test: [(text_embeddings, audio_embeddings), ...],
}
"""

embedding_dct = {
    'train': [],
    'val': [],
    'test': []
}

# Computing embeddings
with torch.no_grad():   
    for i in tqdm(range(len(dataset))):
        split, text, audio_path = dataset[i]

        text_embeddings = clap_model.get_text_embeddings([text])
        audio_embeddings = clap_model.get_audio_embeddings([audio_path], resample=True)

        text_embeddings = text_embeddings.squeeze(0).cpu()
        audio_embeddings = audio_embeddings.squeeze(0).cpu()
        
        # CPUに移して保存
        embedding_dct[split].append(
            (text_embeddings, audio_embeddings)
        )

# Save embeddings
save_path = "../data/audiocaps/clap_embeddings.pt"
torch.save(embedding_dct, save_path)
print(f"Saved CLAP embeddings to {save_path}")
