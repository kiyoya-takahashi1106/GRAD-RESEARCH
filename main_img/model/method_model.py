"""
CLIPの出力に対して、共通-固有分離を用いて共通特徴を抽出するモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import ViTModel


class MethodModel(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, zs_type: str = "common", saved_model_path: str = None):
        super(MethodModel, self).__init__()

        # encoder
        if (hidden_dim == 768):
            self.text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
            self.img_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        elif (hidden_dim == 1024):
            raise ValueError("1024-dim features are not supported.")

        # ===== 重み固定 =====
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.img_encoder.parameters():
            param.requires_grad = False

        # 共通-固有分離用のMLP×4
        self.common_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.common_img_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_img_linear = nn.Linear(hidden_dim, hidden_dim)

        # 再構成用のデコーダ
        self.recon_text_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.recon_img_linear = nn.Linear(hidden_dim*2, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.zs_type = zs_type

        # val用に学習済みモデルをロード
        if (saved_model_path is not None):
            self.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded trained model from {saved_model_path}")


    def forward(self, text_x: torch.Tensor, img_x: torch.Tensor):
        # 共通-固有分離
        common_text = self.common_text_linear(text_x)
        private_text = self.private_text_linear(text_x)
        common_img = self.common_img_linear(img_x)
        private_img = self.private_img_linear(img_x)

        # 再構成
        recon_text = self.recon_text_linear(torch.cat([common_text, private_text], dim=-1))
        recon_img = self.recon_img_linear(torch.cat([common_img, private_img], dim=-1))

        return common_text, common_img, private_text, private_img, recon_text, recon_img
    
    
    # ZS用
    def encode_text(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor):
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        if (self.zs_type == "common"  or  self.zs_type == "cp"):
            text_embedding = self.common_text_linear(text_embedding)
        elif (self.zs_type == "private"  or  self.zs_type == "pc"):
            text_embedding = self.private_text_linear(text_embedding)
        elif (self.zs_type == "concat"):
            common_text = self.common_text_linear(text_embedding)
            private_text = self.private_text_linear(text_embedding)
            text_embedding = torch.cat([common_text, private_text], dim=-1)
        return text_embedding

    def encode_img(self, img_x: torch.Tensor, img_attn_mask: torch.Tensor):
        img_embedding = self.img_encoder(img_x, attention_mask=img_attn_mask).last_hidden_state[:,0,:]
        if (self.zs_type == "common"  or  self.zs_type == "cp"):
            img_embedding = self.common_img_linear(img_embedding)
        elif (self.zs_type == "private"  or  self.zs_type == "pc"):
            img_embedding = self.private_img_linear(img_embedding)
        elif (self.zs_type == "concat"):
            common_img = self.common_img_linear(img_embedding)
            private_img = self.private_img_linear(img_embedding)
            img_embedding = torch.cat([common_img, private_img], dim=-1)
        return img_embedding