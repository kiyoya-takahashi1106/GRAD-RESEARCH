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

from model.clip import Clip
from model.method_model import MethodModel

from datasets.fea_dataset import FeaDataset
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
from utils.utility import set_seed
from utils.utility import compute_contrastive_similarity
from utils.loss import ClapCriterion
from utils.loss import Criterion
from utils.loss import AblationCriterion


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="clip or method or method2 or ablation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str, help="coco")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--dropout_rate", type=float)
    parser.add_argument("--sim_loss_type", type=str, help="cos or cka")
    # hp
    parser.add_argument("--hp_contrastive", type=float)
    parser.add_argument("--hp_sim", type=float)
    parser.add_argument("--hp_cp_diff", type=float)
    parser.add_argument("--hp_pp_diff", type=float)
    parser.add_argument("--hp_recon", type=float)
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.model_type == "clip"):
        model = Clip(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate
        )
    elif (args.model_type == "method"):
        model = MethodModel(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate
        )
    elif (args.model_type == "method2"):
        raise NotImplementedError("Method2Model is not implemented yet.")
        # trained_clap_model_path = f"./saved_models/{args.hidden_dim}/clap_{args.dataset}/best{args.seed}.pth"
        # model = Method2Model(
        #     hidden_dim=args.hidden_dim,
        #     dropout_rate=args.dropout_rate,
        #     saved_model_path=trained_clap_model_path,
        # )
    elif (args.model_type == "ablation"):
        # 各モダリティに1の線形変換、LossはInfoNCE+Lsim
        model = Clip(
            hidden_dim=args.hidden_dim,
            dropout_rate=args.dropout_rate
        )
    
    # TensorBoard Writer設定
    os.makedirs(f"runs/{args.hidden_dim}/{args.model_type}_{args.dataset}_seed{args.seed}", exist_ok=True)
    log_dir = os.path.join("runs", f"{args.hidden_dim}", f"{args.model_type}_{args.dataset}_seed{args.seed}")
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logs will be saved to: {log_dir}")
    
    # モデル全体をGPUに移動
    model = model.to(device)

    scaler = GradScaler()
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    # データセットとデータローダーの準備
    if (args.dataset == "coco"):
        train_dataset = FeaDataset(dataset=args.dataset, split='train', hidden_dim=args.hidden_dim)
        val_dataset = FeaDataset(dataset=args.dataset, split='val', hidden_dim=args.hidden_dim)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print("train dataset size:", len(train_dataset))
    print("val dataset size:", len(val_dataset))

    if (args.model_type == "clip"):
        criterion = ClapCriterion()
    elif (args.model_type == "method"  or  args.model_type == "method2"):
        criterion = Criterion(args.sim_loss_type)
    elif (args.model_type == "ablation"):
        criterion = AblationCriterion(args.sim_loss_type)

    best_val_loss = float('inf')

    
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
            text_embedding, image_embedding = batch
            text_embedding = text_embedding.to(device)       
            image_embedding = image_embedding.to(device)
            text_embedding = torch.squeeze(text_embedding, dim=1)
            image_embedding = torch.squeeze(image_embedding, dim=1)
        
            # forward
            if (args.model_type == "clip"):      
                text_embedding, image_embedding = model(text_embedding, image_embedding)
            elif (args.model_type == "method"  or  args.model_type == "method2"):                         
                common_text, common_image, private_text, private_image, recon_text, recon_image = model(text_embedding, image_embedding)
            elif (args.model_type == "ablation"):
                common_text, common_image = model(text_embedding, image_embedding)
        
            # compute loss
            if (args.model_type == "clip"):
                contractive_loss = criterion.compute_loss(text_embedding, image_embedding)
            elif (args.model_type == "method"  or  args.model_type == "method2"):
                contractive_loss, sim_loss, c2p_text_loss, c2p_image_loss, p2p_loss, recon_text_loss, recon_image_loss  =   criterion.compute_loss(
                                                                                                                                text_embedding, image_embedding,
                                                                                                                                common_text, common_image,
                                                                                                                                private_text, private_image,
                                                                                                                                recon_text, recon_image
                                                                                                                            )
                c2p_loss = (c2p_text_loss + c2p_image_loss) / 2
                recon_loss = (recon_text_loss + recon_image_loss) / 2
            elif (args.model_type == "ablation"):
                contractive_loss, sim_loss = criterion.compute_loss(common_text, common_image)

            # recode loss
            contractive_loss = args.hp_contrastive * contractive_loss
            contractive_loss_lst.append(contractive_loss.item())
            if (args.model_type == "clip"):
                loss = contractive_loss
                loss_lst.append(loss.item())
            elif (args.model_type == "method"  or  args.model_type == "method2"):
                if (args.sim_loss_type == "cos"):
                    sim_loss = args.hp_sim * sim_loss
                    sim_loss2 = sim_loss * (epoch+1) / args.epochs
                elif (args.sim_loss_type == "cka"):
                    sim_loss = args.hp_sim * sim_loss
                sim_loss_lst.append(sim_loss.item())
                c2p_loss = args.hp_cp_diff * c2p_loss
                c2p_loss_lst.append(c2p_loss.item())
                p2p_loss = args.hp_pp_diff * p2p_loss
                p2p_loss_lst.append(p2p_loss.item())
                recon_loss = args.hp_recon * recon_loss
                recon_loss_lst.append(recon_loss.item())
                # 全体loss
                if (args.sim_loss_type == "cos"):
                    if (epoch < 5):
                        loss = contractive_loss + c2p_loss + p2p_loss + recon_loss
                    else:
                        loss = contractive_loss + sim_loss2 + c2p_loss + p2p_loss + recon_loss
                elif (args.sim_loss_type == "cka"):
                    loss = contractive_loss + sim_loss + c2p_loss + p2p_loss + recon_loss
                loss_lst.append(loss.item())
            elif (args.model_type == "ablation"):
                if (args.sim_loss_type == "cos"):
                    sim_loss = args.hp_sim * sim_loss
                elif (args.sim_loss_type == "cka"):
                    sim_loss = args.hp_sim * sim_loss
                sim_loss_lst.append(sim_loss.item())
                # 全体loss
                loss = contractive_loss + sim_loss
                loss_lst.append(loss.item())

            # if ((epoch == 0)):
            #     print("===== INIT =====")
            #     print(f"Contractive Loss: {contractive_loss.item():.6f}")
            #     if (args.model_type == "method"  or  args.model_type == "method2"):
            #         print(f"Sim Loss: {sim_loss.item():.6f}")
            #         print(f"C2P Loss: {c2p_loss.item():.6f}")
            #         print(f"P2P Loss: {p2p_loss.item():.6f}")
            #         print(f"Reconstruction Loss: {recon_loss.item():.6f}")
            #     print("===========================") 

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()

        # loss表示
        print(f"Epoch {epoch}")
        if (args.model_type == "method"  or  args.model_type == "method2"):
            epoch_contractive_loss = sum(contractive_loss_lst) / len(contractive_loss_lst)
            writer.add_scalars('Loss/Train/contractive_Losses', {'Contractive': epoch_contractive_loss}, epoch)
            print(f"Contractive: {epoch_contractive_loss:.6f}")
            epoch_sim_loss = sum(sim_loss_lst) / len(sim_loss_lst)
            writer.add_scalars('Loss/Train/sim_Losses', {'Sim': epoch_sim_loss}, epoch)
            print(f"Sim: {epoch_sim_loss:.6f}")
            epoch_c2p_loss = sum(c2p_loss_lst) / len(c2p_loss_lst)      
            writer.add_scalars('Loss/Train/c2p_Losses', {'C2P': epoch_c2p_loss}, epoch)
            print(f"C2P: {epoch_c2p_loss:.6f}")
            epoch_p2p_loss = sum(p2p_loss_lst) / len(p2p_loss_lst)      
            writer.add_scalars('Loss/Train/p2p_Losses', {'P2P': epoch_p2p_loss}, epoch)
            print(f"P2P: {epoch_p2p_loss:.6f}")
            epoch_recon_loss = sum(recon_loss_lst) / len(recon_loss_lst)
            writer.add_scalars('Loss/Train/recon_Losses', {'Reconstruction': epoch_recon_loss}, epoch)
            print(f"Reconstruction: {epoch_recon_loss:.6f}")
        elif (args.model_type == "ablation"):
            epoch_contractive_loss = sum(contractive_loss_lst) / len(contractive_loss_lst)
            writer.add_scalars('Loss/Train/contractive_Losses', {'Contractive': epoch_contractive_loss}, epoch)
            print(f"Contractive: {epoch_contractive_loss:.6f}")
            epoch_sim_loss = sum(sim_loss_lst) / len(sim_loss_lst)
            writer.add_scalars('Loss/Train/sim_Losses', {'Sim': epoch_sim_loss}, epoch)
            print(f"Sim: {epoch_sim_loss:.6f}")
        epoch_loss = sum(loss_lst) / len(loss_lst)
        writer.add_scalars('Loss/Train/overall_Losses', {'Overall': epoch_loss}, epoch)
        print(f"OverALL: {epoch_loss:.6f}")


        # ===== Evaluation =====
        model.eval()
        val_loss_lst = []

        with torch.no_grad():
            # test_bar = tqdm(test_dataloader, desc=f"Test Epoch {epoch+1}/{args.epochs}", leave=False)
            for batch in val_dataloader:
                text_embedding, image_embedding = batch
                text_embedding = text_embedding.to(device)       
                image_embedding = image_embedding.to(device)
                text_embedding = torch.squeeze(text_embedding, dim=1)
                image_embedding = torch.squeeze(image_embedding, dim=1)

                if (args.model_type == "clip"):
                    text_embedding, image_embedding = model(text_embedding, image_embedding)   
                elif (args.model_type == "method"  or  args.model_type == "method2"):
                    common_text, common_image, private_text, private_image, recon_text, recon_image = model(text_embedding, image_embedding)
                elif (args.model_type == "ablation"):
                    common_text, common_image = model(text_embedding, image_embedding)

                # compute loss
                if (args.model_type == "clip"):
                    contractive_loss = criterion.compute_loss(text_embedding, image_embedding)
                elif (args.model_type == "method"  or  args.model_type == "method2"):
                    contractive_loss, sim_loss, c2p_text_loss, c2p_image_loss, p2p_loss, recon_text_loss, recon_image_loss  =   criterion.compute_loss(
                                                                                                                                    text_embedding, image_embedding,
                                                                                                                                    common_text, common_image,
                                                                                                                                    private_text, private_image,
                                                                                                                                    recon_text, recon_image
                                                                                                                                )
                    c2p_loss = (c2p_text_loss + c2p_image_loss) / 2
                    recon_loss = (recon_text_loss + recon_image_loss) / 2
                elif (args.model_type == "ablation"):
                    contractive_loss, sim_loss = criterion.compute_loss(common_text, common_image)

                # recode loss
                contractive_loss = args.hp_contrastive * contractive_loss
                if (args.model_type == "clip"):
                    loss = contractive_loss
                    val_loss_lst.append(loss.item())
                elif (args.model_type == "method"  or  args.model_type == "method2"):
                    if (args.sim_loss_type == "cos"):
                        sim_loss = args.hp_sim * sim_loss
                    elif (args.sim_loss_type == "cka"):
                        sim_loss = args.hp_sim * sim_loss
                    c2p_loss = args.hp_cp_diff * c2p_loss
                    p2p_loss = args.hp_pp_diff * p2p_loss
                    recon_loss = args.hp_recon * recon_loss
                    # 全体loss
                    loss = contractive_loss + sim_loss + c2p_loss + p2p_loss + recon_loss
                    val_loss_lst.append(loss.item())
                elif (args.model_type == "ablation"):
                    if (args.sim_loss_type == "cos"):
                        sim_loss = args.hp_sim * sim_loss
                    elif (args.sim_loss_type == "cka"):
                        sim_loss = args.hp_sim * sim_loss
                    # 全体loss
                    loss = contractive_loss + sim_loss
                    val_loss_lst.append(loss.item())

        epoch_val_loss = sum(val_loss_lst) / len(val_loss_lst)
        writer.add_scalars('Loss/Val', {'val_loss': epoch_val_loss}, epoch)
        print(f"Val Loss: {epoch_val_loss:.6f}")
        
        
        # モデル保存
        # if (epoch == args.epochs - 1):
        if (epoch_val_loss <= best_val_loss):
            best_val_loss = epoch_val_loss
            os.makedirs(os.path.dirname(f"saved_models/{args.hidden_dim}/{args.model_type}_{args.dataset}/"), exist_ok=True)
            best_model_path = (
                f"saved_models/{args.hidden_dim}/{args.model_type}_{args.dataset}/"
                f"best{args.seed}.pth"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"We've saved the new model (Val Loss: {best_val_loss:.4f})")
        print("----------------------------------------------------------------------------")

    print(f"Best Val Loss: {best_val_loss:.4f}")
    writer.close()
    return



if (__name__ == "__main__"):
    _args = args()
    for arg in vars(_args):
        print(f"{arg}: {getattr(_args, arg)}")
    set_seed(_args.seed)
    train(_args)