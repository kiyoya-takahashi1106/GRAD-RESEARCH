"""
CLAP ESC-50 TEST

The output:
ESC50 Accuracy: 0.826%
"""

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

from datasets.esc50_dataset import ESC50Dataset


# Load dataset
root_path = "../data"
dataset = ESC50Dataset(root=root_path, download=True)

# Load and initialize CLAP
clap_model = CLAP(version = '2022', use_cuda=True)

# Computing text embeddings
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())


y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))