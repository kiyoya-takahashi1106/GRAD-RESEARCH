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

from model.clap import Clap
from model.method_model import MethodModel
from transformers import RobertaTokenizerFast, Wav2Vec2Processor

from datasets.esc50_dataset import ESC50Dataset


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--saved_model_path", type=str)
    args = parser.parse_args()
    return args


def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.model_type == "clap"):
        model = Clap(
            dropout_rate=args.dropout_rate,
            saved_model_path=args.saved_model_path
        )
    elif (args.model_type == "method"):
        model = MethodModel(
            dropout_rate=args.dropout_rate,
            saved_model_path=args.saved_model_path
        )
        
    model = model.to(device)

    text_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    test_dataset = ESC50Dataset(text_tokenizer=text_tokenizer, audio_processor=audio_processor, split="None")

    # ===== 1. クラス側テキストの埋め込み =====
    text_input_ids = test_dataset.input_ids.to(device)        # (C, L)
    text_attn_masks = test_dataset.attn_masks.to(device)      # (C, L)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_input_ids, text_attn_masks)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)


    # ===== 2. 各音声サンプルの zero-shot 推論 =====
    y_preds, y_labels = [], []

    for i in tqdm(range(len(test_dataset)), desc="Zero-shot eval"):
        audio_x, attn_mask, one_hot_target = test_dataset[i]

        audio_x = audio_x.to(device)
        attn_mask = attn_mask.to(device)

        with torch.no_grad():
            audio_embedding = model.encode_audio(audio_x, attn_mask)
            audio_embedding = F.normalize(audio_embedding, p=2, dim=-1)

            # audio (1,D) × text.T (D,C) = (1,C)
            similarity = audio_embedding @ text_embeddings.T

            # softmax でクラス確率に
            probs = F.softmax(similarity, dim=-1).detach().cpu().numpy()  # (1, C)

        # one-hot も numpy に
        one_hot_np = one_hot_target.detach().cpu().numpy()[None, :]       # (1, C)

        y_preds.append(probs)
        y_labels.append(one_hot_np)

    # ===== 3. 精度計算 =====
    y_labels = np.concatenate(y_labels, axis=0)  # (N, C)
    y_preds = np.concatenate(y_preds, axis=0)    # (N, C)

    acc = accuracy_score(np.argmax(y_labels, axis=1),
                         np.argmax(y_preds, axis=1))
    print(f"{args.dataset} Zero-shot Accuracy: {acc:.4f}")

    return acc



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    val(_args)