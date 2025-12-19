"""
CLAP ESC-50 TEST (Cosine similarity version)

Outputs:
- ESC50 Accuracy (using cosine similarity)
- positive mean cosine similarity
- negative mean cosine similarity
"""

import torch
import torch.nn.functional as F
print("CUDA available:", torch.cuda.is_available())

from msclap import CLAP
from tqdm import tqdm

import numpy as np
from sklearn.metrics import accuracy_score

from datasets.esc50_dataset import ESC50Dataset


# Load dataset
root_path = "../data"
dataset = ESC50Dataset(root=root_path, download=True)

# Load and initialize CLAP
clap_model = CLAP(version='2022', use_cuda=True)

# Computing text embeddings
prompt = 'this is the sound of '
class_texts = [prompt + x for x in dataset.classes]
text_embeddings = clap_model.get_text_embeddings(class_texts)

# msclap may return numpy -> torch
if isinstance(text_embeddings, np.ndarray):
    text_embeddings = torch.from_numpy(text_embeddings)

# L2 normalize (for true cosine similarity)
text_embeddings = F.normalize(text_embeddings, dim=-1)   # [C, D]


# Buffers
y_preds, y_labels = [], []
pos_sims = []
neg_sims = []

# Computing audio embeddings
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset[i]
    gt_class = int(torch.argmax(one_hot_target).item())

    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    if isinstance(audio_embeddings, np.ndarray):
        audio_embeddings = torch.from_numpy(audio_embeddings)

    # audio_embeddings is typically [1, D]
    audio_embeddings = F.normalize(audio_embeddings, dim=-1)

    a = audio_embeddings.squeeze(0)  # [D]

    # True cosine similarity: [C] in [-1, 1]
    sim = torch.matmul(a, text_embeddings.T).detach().cpu()

    # Accuracy (use cosine sim as logits)
    y_pred = F.softmax(sim.unsqueeze(0), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy().reshape(1, -1))

    # Cosine stats
    pos_sims.append(sim[gt_class].item())
    neg_mask = torch.ones(len(sim), dtype=torch.bool)
    neg_mask[gt_class] = False
    neg_sims.extend(sim[neg_mask].tolist())


# Accuracy
y_labels = np.concatenate(y_labels, axis=0)
y_preds = np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

# Cosine stats
pos_sims = np.array(pos_sims, dtype=np.float32)
neg_sims = np.array(neg_sims, dtype=np.float32)

print(f'ESC50 Accuracy {acc:.4f}')
print('Cosine similarity (L2-normalized dot):')
print(f'  positive mean = {pos_sims.mean():.4f}')
print(f'  negative mean = {neg_sims.mean():.4f}')