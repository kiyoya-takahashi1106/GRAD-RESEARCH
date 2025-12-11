import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())

import os
import sys
import argparse
from tqdm.auto import tqdm
from functools import partial

from model.clap import Clap
from model.method_model import MethodModel

from datasets.fea_dataset import FeaDataset
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.utility import set_seed
from utils.utility import compute_contrastive_similarity
from utils.loss import ClapCriterion
from utils.loss import Criterion


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="clap or method")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str, help="audiocaps or fsd50k or mix")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--dropout_rate", type=float)
    # hp
    parser.add_argument("--hp_contrastive", type=float)
    parser.add_argument("--hp_sim", type=float)
    parser.add_argument("--hp_diff", type=float)
    parser.add_argument("--hp_recon", type=float)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.model_type == "clap"):
        model = Clap(
            dropout_rate=args.dropout_rate
        )
    elif (args.model_type == "method"):
        model = MethodModel(
            dropout_rate=args.dropout_rate
        )
    
    # TensorBoard Writer設定
    os.makedirs(f"runs/train/{args.model_type}_{args.dataset}", exist_ok=True)
    log_dir = os.path.join("runs", f"train/{args.model_type}_{args.dataset}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # モデル全体をGPUに移動 
    model = model.to(device)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # データセットとデータローダーの準備
    if (args.dataset == "audiocaps"  or  args.dataset == "fsd50k"):
        train_dataset = FeaDataset(dataset=args.dataset, split='train')
        test_dataset = FeaDataset(dataset=args.dataset, split='val')
    elif (args.dataset == "mix"):
        train_dataset = ConcatDataset([FeaDataset(dataset="audiocaps", split='train'), FeaDataset(dataset="fsd50k", split='train')])
        test_dataset = ConcatDataset([FeaDataset(dataset="audiocaps", split='val'), FeaDataset(dataset="fsd50k", split='val')])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print("train dataset size:", len(train_dataset))
    print("test dataset size:", len(test_dataset))

    if (args.model_type == "clap"):
        criterion = ClapCriterion()
    elif (args.model_type == "method"):
        criterion = Criterion()

    best_contractive = float('inf')

    
    for epoch in range(args.epochs):
        # ===== Training =====
        model.train()
        contractive_loss_lst = []
        sim_loss_lst = []
        c2p_loss_lst = []
        p2p_loss_lst = []
        recon_loss_lst = []
        loss_lst = []

        # train_bar = tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{args.epochs}", leave=False)
        for i, batch in enumerate(train_dataloader):
            text_embedding, audio_embedding = batch
            text_embedding = text_embedding.to(device)       
            audio_embedding = audio_embedding.to(device)
            text_embedding = torch.squeeze(text_embedding, dim=1)
            audio_embedding = torch.squeeze(audio_embedding, dim=1)

            # forward
            if (args.model_type == "clap"):      
                text_embedding, audio_embedding = model(text_embedding, audio_embedding)
            elif (args.model_type == "method"):                         
                common_text, common_audio, private_text, private_audio, recon_text, recon_audio = model(text_embedding, audio_embedding)

            # compute loss
            if (args.model_type == "clap"):
                contractive_loss = criterion.compute_loss(text_embedding, audio_embedding)
            elif (args.model_type == "method"):
                contractive_loss, sim_loss, c2p_text_loss, c2p_audio_loss, p2p_loss, recon_text_loss, recon_audio_loss  =   criterion.compute_loss(
                                                                                                                                text_embedding, audio_embedding,
                                                                                                                                common_text, common_audio,
                                                                                                                                private_text, private_audio,
                                                                                                                                recon_text, recon_audio
                                                                                                                            )
                recon_loss = (recon_text_loss + recon_audio_loss) / 2
                c2p_loss = (c2p_text_loss + c2p_audio_loss) / 2

            # recode loss
            contractive_loss = args.hp_contrastive * contractive_loss
            contractive_loss_lst.append(contractive_loss.item())
            if (args.model_type == "clap"):
                loss = contractive_loss
                loss_lst.append(loss.item())
            elif (args.model_type == "method"):
                sim_loss = args.hp_sim * sim_loss
                sim_loss_lst.append(sim_loss.item())
                c2p_loss = args.hp_diff * c2p_loss
                c2p_loss_lst.append(c2p_loss.item())
                p2p_loss = args.hp_diff * p2p_loss
                p2p_loss_lst.append(p2p_loss.item())
                recon_loss = args.hp_recon * recon_loss
                recon_loss_lst.append(recon_loss.item())
                # 全体loss
                loss = contractive_loss + sim_loss + c2p_loss + p2p_loss + recon_loss
                # loss = contractive_loss + sim_loss + p2p_loss + recon_loss
                loss_lst.append(loss.item())

            if ((epoch == 0)):
                print("===== INIT =====")
                print(f"Contractive Loss: {contractive_loss.item():.6f}")
                if (args.model_type == "method"):
                    print(f"Sim Loss: {sim_loss.item():.6f}")
                    print(f"C2P Loss: {c2p_loss.item():.6f}")
                    print(f"P2P Loss: {p2p_loss.item():.6f}")
                    print(f"Reconstruction Loss: {recon_loss.item():.6f}")
                print("===========================") 

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # loss表示
        print(f"Epoch {epoch}")
        if (args.model_type == "method"):
            epoch_contractive_loss = sum(contractive_loss_lst) / len(contractive_loss_lst)
            writer.add_scalars('Loss/Train/Epoch/contractive_Losses', {'Contractive': epoch_contractive_loss}, epoch)
            print(f"Contractive: {epoch_contractive_loss:.6f}")
            epoch_sim_loss = sum(sim_loss_lst) / len(sim_loss_lst)
            writer.add_scalars('Loss/Train/Epoch/sim_Losses', {'Sim': epoch_sim_loss}, epoch)
            print(f"Sim: {epoch_sim_loss:.6f}")
            epoch_c2p_loss = sum(c2p_loss_lst) / len(c2p_loss_lst)      
            writer.add_scalars('Loss/Train/Epoch/c2p_Losses', {'C2P': epoch_c2p_loss}, epoch)
            print(f"C2P: {epoch_c2p_loss:.6f}")
            epoch_p2p_loss = sum(p2p_loss_lst) / len(p2p_loss_lst)      
            writer.add_scalars('Loss/Train/Epoch/p2p_Losses', {'P2P': epoch_p2p_loss}, epoch)
            print(f"P2P: {epoch_p2p_loss:.6f}")
            epoch_recon_loss = sum(recon_loss_lst) / len(recon_loss_lst)
            writer.add_scalars('Loss/Train/Epoch/recon_Losses', {'Reconstruction': epoch_recon_loss}, epoch)
            print(f"Reconstruction: {epoch_recon_loss:.6f}")
        epoch_loss = sum(loss_lst) / len(loss_lst)
        writer.add_scalars('Loss/Train/Epoch/overall_Losses', {'Overall': epoch_loss}, epoch)
        print(f"OverALL: {epoch_loss:.6f}")


        # ===== Evaluation =====
        model.eval()
        test_contractive_lst = []

        with torch.no_grad():
            # test_bar = tqdm(test_dataloader, desc=f"Test Epoch {epoch+1}/{args.epochs}", leave=False)
            for batch in test_dataloader:
                text_embedding, audio_embedding = batch
                text_embedding = text_embedding.to(device)       
                audio_embedding = audio_embedding.to(device)
                text_embedding = torch.squeeze(text_embedding, dim=1)
                audio_embedding = torch.squeeze(audio_embedding, dim=1)

                if (args.model_type == "clap"):
                    text_embedding, audio_embedding = model(text_embedding, audio_embedding)   
                elif (args.model_type == "method"):
                    common_text, common_audio, _, _, _, _ = model(text_embedding, audio_embedding)
                    text_embedding = common_text
                    audio_embedding = common_audio

                contractive = compute_contrastive_similarity(text_embedding, audio_embedding)
                contractive = args.hp_contrastive * contractive
                test_contractive_lst.append(contractive.item())

        epoch_test_contractive = sum(test_contractive_lst) / len(test_contractive_lst)
        writer.add_scalars('Loss/Test/Epoch/Contrastive', {'Contrastive': epoch_test_contractive}, epoch)
        print(f"Test Contrastive: {epoch_test_contractive:.6f}")
        
        # モデル保存
        if (epoch_test_contractive < best_contractive):
            best_contractive = epoch_test_contractive
            os.makedirs(
                f"saved_models/train/{args.model_type}_{args.dataset}/", exist_ok=True
            )
            # best_model_path = (
            #     f"saved_models/train/{args.model_type}_{args.dataset}/"
            #     f"epoch{epoch}.pth"
            # )
            best_model_path = (
                f"saved_models/train/{args.model_type}_{args.dataset}/"
                f"best.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"We've saved the new model (Contrastive: {best_contractive:.4f})")
        print("----------------------------------------------------------------------------")

    print(f"Best Contrastive: {best_contractive:.4f}")
    writer.close()
    return



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    set_seed(_args.seed)
    train(_args)