import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from msclap import CLAP

from model.model import Model
from datasets.esc50_dataset import ESC50Dataset


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="esc50")
    parser.add_argument("--dropout_rate", type=float, default=0.0)
    parser.add_argument("--saved_model_path", type=str, required=True)
    return parser.parse_args()


def val(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Models
    clap_model = CLAP(version="2022", use_cuda=True)
    model = Model(
        dropout_rate=args.dropout_rate,
        saved_model_path=args.saved_model_path
    ).to(device)
    model.eval()

    # Dataset
    root_path = "../data"
    val_dataset = ESC50Dataset(root=root_path, download=True)

    # Text embeddings (class prompts)
    prompt = "this is the sound of "
    class_texts = [prompt + x for x in val_dataset.classes]

    text_embeddings = clap_model.get_text_embeddings(class_texts)
    if isinstance(text_embeddings, np.ndarray):
        text_embeddings = torch.from_numpy(text_embeddings)
    text_embeddings = text_embeddings.to(device)

    with torch.no_grad():
        common_text = model.extract_common_text(text_embeddings)  # [C, D] or [C, ...]
        # L2 normalize for true cosine similarity
        common_text = F.normalize(common_text, dim=-1)

    # Buffers
    y_preds, y_labels = [], []
    pos_sims = []
    neg_sims = []

    # Evaluation loop
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset))):
            x, _, one_hot_target = val_dataset[i]
            gt_class = int(torch.argmax(one_hot_target).item())

            audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
            if isinstance(audio_embeddings, np.ndarray):
                audio_embeddings = torch.from_numpy(audio_embeddings)
            audio_embeddings = audio_embeddings.to(device)

            common_audio = model.extract_common_audio(audio_embeddings)  # [1, D] or [D]
            common_audio = F.normalize(common_audio, dim=-1)

            # True cosine similarity (no CLAP logit_scale)
            # Handle shapes:
            # common_audio: [1, D] (from get_audio_embeddings([x])) is typical
            if common_audio.dim() == 2:
                a = common_audio.squeeze(0)  # [D]
            else:
                a = common_audio  # [D]
            # common_text: [C, D]
            sim = torch.matmul(a, common_text.T).detach().cpu()  # [C], in [-1, 1]

            # Accuracy (use cosine sim logits directly)
            y_pred = F.softmax(sim.unsqueeze(0), dim=1).numpy()
            y_preds.append(y_pred)
            y_labels.append(one_hot_target.cpu().numpy().reshape(1, -1))

            # Cosine similarity stats
            pos_sims.append(sim[gt_class].item())
            # vectorized negatives
            neg_mask = torch.ones(len(sim), dtype=torch.bool)
            neg_mask[gt_class] = False
            neg_sims.extend(sim[neg_mask].tolist())

    # Accuracy
    y_labels = np.concatenate(y_labels, axis=0)
    y_preds = np.concatenate(y_preds, axis=0)
    acc = accuracy_score(np.argmax(y_labels, axis=1),
                         np.argmax(y_preds, axis=1))

    # Cosine similarity stats
    pos_sims = np.array(pos_sims, dtype=np.float32)
    neg_sims = np.array(neg_sims, dtype=np.float32)

    print(f"\n{args.dataset} Accuracy: {acc:.4f}")
    print("Cosine similarity (L2-normalized dot):")
    print(f"  positive mean = {pos_sims.mean():.4f}")
    print(f"  negative mean = {neg_sims.mean():.4f}")


if __name__ == "__main__":
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    val(_args)