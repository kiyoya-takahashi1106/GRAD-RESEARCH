"""
CLAPの再現モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import Wav2Vec2Model


class Clap(nn.Module):
    def __init__(self, dropout_rate: float, hidden_dim: int = 768, saved_model_path: str = None):
        super(Clap, self).__init__()        

        # encoder
        self.text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # ===== 重み固定 =====
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # projection*2
        self.text_projection = nn.Linear(hidden_dim, hidden_dim)
        self.audio_projection = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # val用に学習済みモデルをロード
        if (saved_model_path is not None):
            self.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded trained model from {saved_model_path}")


    def forward(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor, audio_x: torch.Tensor, audio_attn_mask: torch.Tensor):
        # enocode
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        audio_embedding = self.audio_encoder(audio_x, attention_mask=audio_attn_mask).last_hidden_state.mean(dim=1)

        # 共通-固有分離
        text_embedding = self.text_projection(text_embedding)
        audio_embedding = self.audio_projection(audio_embedding)

        return text_embedding, audio_embedding


    def encode_text(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor):
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        text_embedding = self.text_projection(text_embedding)
        return text_embedding
    

    def encode_audio(self, audio_x: torch.Tensor, audio_attn_mask: torch.Tensor):
        audio_embedding = self.audio_encoder(audio_x, attention_mask=audio_attn_mask).last_hidden_state.mean(dim=1)
        audio_embedding = self.audio_projection(audio_embedding)
        return audio_embedding