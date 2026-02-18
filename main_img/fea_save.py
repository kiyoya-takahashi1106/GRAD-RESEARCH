import torch
import torch.nn as nn
import torch.nn.functional as F
print("CUDA available:", torch.cuda.is_available())

from tqdm import tqdm

import os
import sys
import numpy as np
import argparse
from functools import partial

from transformers import RobertaTokenizerFast
from transformers import AutoModel
from transformers import ViTModel, ViTImageProcessor

from datasets.coco_dataset import COCODataset
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.utility import set_seed
from utils.collate_fn_img import collate_fn_img


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, help="coco")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--hidden_dim", type=int, default=768)
args = parser.parse_args()

set_seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
if (args.hidden_dim == 768):
    print("Using 768-dim features")
    text_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False).to(device)
    img_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
elif (args.hidden_dim == 1024):
    raise ValueError("1024-dim features are not supported.")


text_encoder.eval()
img_encoder.eval()

# freeze parameters
for param in text_encoder.parameters():
    param.requires_grad = False
for param in img_encoder.parameters():
    param.requires_grad = False

# Load dataset
if (args.dataset == "coco"):
    dataset = COCODataset(split="all")
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn_img, text_tokenizer=text_tokenizer, img_processor=img_processor))


"""
{
    train: [(text_fea, img_fea), ...],
      val: [(text_fea, img_fea), ...],
     test: [(text_fea, img_fea), ...],
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
        splits, text_x, text_attn_mask, img_x, img_attn_mask = batch
        text_x = text_x.to(device)
        text_attn_mask = text_attn_mask.to(device)
        img_x = img_x.to(device)
        # img_attn_mask = img_attn_mask.to(device)

        text_embedding = text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        img_embedding = img_encoder(img_x, attention_mask=img_attn_mask).last_hidden_state.mean(dim=1)
        
        # save embedding
        for i in range(len(batch[0])):
            split = splits[i]
            text_embedding_i = text_embedding[i].unsqueeze(0).cpu()
            img_embedding_i = img_embedding[i].unsqueeze(0).cpu()
            fea_dct[split].append(
                (text_embedding_i, img_embedding_i)
            )

# Save fea
save_path = f"../data/{args.dataset}/fea{args.hidden_dim}_imageNet.pt"
torch.save(fea_dct, save_path)
print(f"Saved fea to {save_path}")