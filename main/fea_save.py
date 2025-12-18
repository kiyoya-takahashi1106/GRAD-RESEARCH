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

from transformers import RobertaTokenizerFast, Wav2Vec2Processor
from transformers import AutoModel
from transformers import Wav2Vec2Model

from datasets.audiocaps_dataset import AudioCapsDataset
from datasets.fsd50k_dataset import FSD50KDataset
from datasets.clotho_dataset import ClothoDataset
from torch.utils.data import DataLoader

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.utility import set_seed
from utils.collate_fn import collate_fn


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dataset", type=str, help="audiocaps, fsd50k, clotho")
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()

set_seed(args.seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False).to(device)
audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(device)
text_encoder.eval()
audio_encoder.eval()
text_tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

# freeze parameters
for param in text_encoder.parameters():
    param.requires_grad = False
for param in audio_encoder.parameters():
    param.requires_grad = False

# Load dataset
if (args.dataset == "audiocaps"):
    dataset = AudioCapsDataset(split="all")
elif (args.dataset == "fsd50k"):
    dataset = FSD50KDataset(split="all")
elif (args.dataset == "clotho"):
    dataset = ClothoDataset(split="all")
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate_fn, text_tokenizer=text_tokenizer, audio_processor=audio_processor))


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
        splits, text_x, text_attn_mask, audio_x, audio_attn_mask = batch
        text_x = text_x.to(device)
        text_attn_mask = text_attn_mask.to(device)
        audio_x = audio_x.to(device)
        audio_attn_mask = audio_attn_mask.to(device)

        text_embedding = text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        audio_embedding = audio_encoder(audio_x, attention_mask=audio_attn_mask).last_hidden_state.mean(dim=1)
        
        # save embedding
        for i in range(len(batch[0])):
            split = splits[i]
            text_embedding_i = text_embedding[i].unsqueeze(0).cpu()
            audio_embedding_i = audio_embedding[i].unsqueeze(0).cpu()
            fea_dct[split].append(
                (text_embedding_i, audio_embedding_i)
            )

# Save fea
save_path = f"../data/{args.dataset}/fea.pt"
torch.save(fea_dct, save_path)
print(f"Saved fea to {save_path}")