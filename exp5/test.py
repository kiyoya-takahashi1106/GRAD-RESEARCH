import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from msclap import CLAP

from model.model import Model

from datasets.esc50_dataset import ESC50Dataset


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--saved_model_path", type=str)
    args = parser.parse_args()
    return args


def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clap_model = CLAP(version = '2022', use_cuda=True)
    model = Model(
        dropout_rate=args.dropout_rate,
        saved_model_path=args.saved_model_path
    )
    model = model.to(device)

    # データセットとデータローダーの準備
    root_path = "../data"
    val_dataset = ESC50Dataset(root=root_path, download=True)

    # Computing text embeddings
    prompt = 'this is the sound of '
    y = [prompt + x for x in val_dataset.classes]
    text_embeddings = clap_model.get_text_embeddings(y)
    common_text = model.extract_common_text(text_embeddings)

    # Computing audio embeddings
    y_preds, y_labels = [], []
    for i in tqdm(range(len(val_dataset))):
        x, _, one_hot_target = val_dataset.__getitem__(i)
        audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
        common_audio = model.extract_common_audio(audio_embeddings)
        similarity = clap_model.compute_similarity(common_audio, common_text)
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())


    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
    print(f'{args.dataset} Accuracy {acc}')

    return



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    val(_args)