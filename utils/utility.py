import torch
import torch.nn.functional as F

import os
import random
import numpy as np


def set_seed(seed :int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_similarity(text_embeddings: torch.Tensor, audio_embeddings: torch.Tensor):
    """
    text_embeddings: (batch_size, dim)
    audio_embeddings: (batch_size, dim)
    """
    text_embeddings_norm = F.normalize(text_embeddings, p=2, dim=-1)
    audio_embeddings_norm = F.normalize(audio_embeddings, p=2, dim=-1)
    sim = torch.sum(text_embeddings_norm * audio_embeddings_norm, dim=-1)
    sim = sim.mean()
    return sim


def compute_contrastive_similarity(text_embeddings: torch.Tensor, audio_embeddings: torch.Tensor):
    """
    text_embeddings: (batch_size, dim)
    audio_embeddings: (batch_size, dim)
    """
    batch_size = text_embeddings.size(0)
    temperature = 0.1

    # 正規化
    common_text_norm = F.normalize(text_embeddings, p=2, dim=1)
    common_audio_norm = F.normalize(audio_embeddings, p=2, dim=1)

    # 類似度計算
    logits = torch.matmul(common_text_norm, common_audio_norm.T) / temperature

    labels = torch.arange(batch_size).to(text_embeddings.device)

    loss_text_to_audio = F.cross_entropy(logits, labels)
    loss_audio_to_text = F.cross_entropy(logits.T, labels)

    loss = (loss_text_to_audio + loss_audio_to_text) / 2
    return loss