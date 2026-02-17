"""
CLAP系の重みに対して、共通-固有分離を用いて共通表現を抽出するモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import Wav2Vec2Model


class Method2Model(nn.Module):
    def __init__(self, hidden_dim: int, dropout_rate: float, saved_model_path: str, zs_type: str = "common"):
        super(Method2Model, self).__init__()        

        # encoder
        if (hidden_dim == 768):
            self.text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
            self.audio_encoder = Wav2Vec2Model.from_pretrained("ALM/wav2vec2-base-audioset")
        elif (hidden_dim == 1024):
            self.text_encoder = AutoModel.from_pretrained("roberta-large", add_pooling_layer=False)
            self.audio_encoder = Wav2Vec2Model.from_pretrained("ALM/wav2vec2-large-audioset")

        # ===== Encoder重み固定 =====
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # projection*2
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.audio_projection = nn.Linear(hidden_dim, hidden_dim)

        # ===== projection重み固定 =====
        for param in self.text_projection.parameters():
            param.requires_grad = False
        for param in self.audio_projection.parameters():
            param.requires_grad = False


        # 共通-固有分離用のMLP×4
        self.common_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.common_audio_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_audio_linear = nn.Linear(hidden_dim, hidden_dim)

        # 再構成用のデコーダ
        self.recon_text_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.recon_audio_linear = nn.Linear(hidden_dim*2, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.zs_type = zs_type

        # 学習済みモデルをロード
        state_dict = torch.load(saved_model_path)
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded trained model from {saved_model_path}")


    def forward(self, text_embedding: torch.Tensor, audio_embedding: torch.Tensor):
        # 共通空間への射影
        text_embedding = self.text_projection(text_embedding)
        audio_embedding = self.audio_projection(audio_embedding)

        # 共通-固有分離
        common_text = self.common_text_linear(text_embedding)
        private_text = self.private_text_linear(text_embedding)
        common_audio = self.common_audio_linear(audio_embedding)
        private_audio = self.private_audio_linear(audio_embedding)

        # 再構成
        recon_text = self.recon_text_linear(torch.cat([common_text, private_text], dim=-1))
        recon_audio = self.recon_audio_linear(torch.cat([common_audio, private_audio], dim=-1))

        return common_text, common_audio, private_text, private_audio, recon_text, recon_audio


    # ZS用
    def encode_text(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor):
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        text_embedding = self.text_projection(text_embedding)
        if (self.zs_type == "common"):
            text_embedding = self.common_text_linear(text_embedding)
        elif (self.zs_type == "private"):
            text_embedding = self.private_text_linear(text_embedding)
        elif (self.zs_type == "add"):
            text_embedding = self.common_text_linear(text_embedding) + self.private_text_linear(text_embedding)
        elif (self.zs_type == "concat"):
            common_text = self.common_text_linear(text_embedding)
            private_text = self.private_text_linear(text_embedding)
            text_embedding = torch.cat([common_text, private_text], dim=-1)
        return text_embedding    

    def encode_audio(self, audio_x: torch.Tensor, audio_attn_mask: torch.Tensor):
        audio_embedding = self.audio_encoder(audio_x, attention_mask=audio_attn_mask).last_hidden_state.mean(dim=1)
        audio_embedding = self.audio_projection(audio_embedding)
        if (self.zs_type == "common"):
            audio_embedding = self.common_audio_linear(audio_embedding)
        elif (self.zs_type == "private"):
            audio_embedding = self.private_audio_linear(audio_embedding)        
        elif (self.zs_type == "add"):
            audio_embedding = self.common_audio_linear(audio_embedding) + self.private_audio_linear(audio_embedding)
        elif (self.zs_type == "concat"):
            common_audio = self.common_audio_linear(audio_embedding)
            private_audio = self.private_audio_linear(audio_embedding)
            audio_embedding = torch.cat([common_audio, private_audio], dim=-1)
        return audio_embedding