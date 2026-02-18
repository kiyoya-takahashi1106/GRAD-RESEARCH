"""
CLIPの再現モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import ViTModel


class Clip(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, saved_model_path: str = None):
        super(Clip, self).__init__()        

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

        # projection*2
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.img_projection = nn.Linear(hidden_dim, hidden_dim)

        # val用に学習済みモデルをロード
        if (saved_model_path is not None):
            self.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded trained model from {saved_model_path}")


    def forward(self, text_x: torch.Tensor, img_x: torch.Tensor):
        # 共通空間への射影
        text_embedding = self.text_projection(text_x)
        img_embedding = self.img_projection(img_x)
        return text_embedding, img_embedding


    # ZS用
    def encode_text(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor):
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        text_embedding = self.text_projection(text_embedding)
        return text_embedding    

    def encode_img(self, img_x: torch.Tensor, attn_mask: torch.Tensor = None):
        img_embedding = self.img_encoder(img_x, attention_mask=attn_mask).last_hidden_state[:,0,:]
        img_embedding = self.img_projection(img_embedding)
        return img_embedding