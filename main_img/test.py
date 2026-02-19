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

from transformers import RobertaTokenizerFast
from transformers import ViTImageProcessor

from model.clip import Clip
from model.method_model import MethodModel

from datasets.caltech101_dataset import Caltech101Dataset
from datasets.oxford_pet_dataset import OxfordPetDataset


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="clip or method or method2 or ablation")
    parser.add_argument("--dataset", type=str, help="caltech101")
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--zs_type", type=str, help="common or private or cp or pc or concat")
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--saved_model_path", type=str)
    args = parser.parse_args()
    return args


def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (args.model_type == "clip"  or  args.model_type == "ablation"):
        model = Clip(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            saved_model_path=args.saved_model_path
        )
    elif (args.model_type == "method"):
        model = MethodModel(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate,
            zs_type=args.zs_type,
            saved_model_path=args.saved_model_path
        )
    elif (args.model_type == "method2"):
        raise ValueError("method2 is not supported.")
        
    model = model.to(device)

    # Load tokenizers and processors
    if (args.hidden_dim == 768):
        text_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    elif (args.hidden_dim == 1024):
        raise ValueError("1024-dim features are not supported.")

    if (args.dataset == "caltech101"):
        test_dataset = Caltech101Dataset(text_tokenizer=text_tokenizer, img_processor=img_processor, split="None")
    elif (args.dataset == "oxford_pet"):
        test_dataset = OxfordPetDataset(text_tokenizer=text_tokenizer, img_processor=img_processor, split="None")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f"Test dataset size: {len(test_dataset)}")


    # ===== 1. クラス側テキストの埋め込み =====
    text_input_ids = test_dataset.input_ids.to(device)        # (C, L)
    text_attn_masks = test_dataset.attn_masks.to(device)      # (C, L)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_input_ids, text_attn_masks)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)


    # ===== 2. 各画像サンプルの zero-shot 推論 =====
    y_preds, y_labels = [], []
    pos_sims, neg_sims = [], []

    for i in range(len(test_dataset)):
        img_x, attn_mask, one_hot_target = test_dataset[i]
        gt_class = int(torch.argmax(one_hot_target).item())

        img_x = img_x.to(device)

        with torch.no_grad():
            img_embedding = model.encode_img(img_x, attn_mask)
            img_embedding = F.normalize(img_embedding, p=2, dim=-1)

            # img (1,D) × text.T (D,C) = (1,C)
            similarity = img_embedding @ text_embeddings.T
            similarity = similarity.squeeze(0)  # (C,)

            # softmax でクラス確率に
            probs = F.softmax(similarity.unsqueeze(0), dim=-1).detach().cpu().numpy()  # (1, C)

        # one-hot も numpy に
        one_hot_np = one_hot_target.detach().cpu().numpy()[None, :]  # (1, C)

        y_preds.append(probs)
        y_labels.append(one_hot_np)

        # Cosine similarity stats
        pos_sims.append(similarity[gt_class].item())
        # vectorized negatives
        neg_mask = torch.ones(len(similarity), dtype=torch.bool)
        neg_mask[gt_class] = False
        neg_sims.extend(similarity[neg_mask].detach().cpu().tolist())

    # ===== 3. 精度計算 =====
    y_labels = np.concatenate(y_labels, axis=0)  # (N, C)
    y_preds = np.concatenate(y_preds, axis=0)    # (N, C)

    acc = accuracy_score(np.argmax(y_labels, axis=1),
                         np.argmax(y_preds, axis=1))
    
    # Cosine similarity stats
    pos_sims = np.array(pos_sims, dtype=np.float32)
    neg_sims = np.array(neg_sims, dtype=np.float32)

    print(f"{args.dataset} Zero-shot Accuracy: {acc:.4f}")
    print("Cosine similarity (L2-normalized dot):")
    print(f"  positive = {pos_sims.mean():.4f}±{pos_sims.std():.4f}")
    print(f"  negative = {neg_sims.mean():.4f}±{neg_sims.std():.4f}")
    print(f"\n")

    return acc



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    val(_args)