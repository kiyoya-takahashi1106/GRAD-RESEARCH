"""
CLAPの出力に対して、共通-固有分離を用いて共通特徴を抽出するモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel
from transformers import Wav2Vec2Model


class MethodModel(nn.Module):
    def __init__(self, dropout_rate: float, hidden_dim: int = 768, saved_model_path: str = None):
        super(MethodModel, self).__init__()

        # encoder
        self.text_encoder = AutoModel.from_pretrained("roberta-base", add_pooling_layer=False)
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        # ===== 重み固定 =====
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        for param in self.audio_encoder.parameters():
            param.requires_grad = False

        # 共通-固有分離用のMLP×4
        self.common_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.common_audio_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_audio_linear = nn.Linear(hidden_dim, hidden_dim)

        # 固有同士の分類をするdiscriminator
        # self.discriminator = nn.Linear(hidden_dim, 1)

        # 再構成用のデコーダ
        self.recon_text_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.recon_audio_linear = nn.Linear(hidden_dim*2, hidden_dim)

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
        common_text = self.common_text_linear(text_embedding)
        private_text = self.private_text_linear(text_embedding)
        common_audio = self.common_audio_linear(audio_embedding)
        private_audio = self.private_audio_linear(audio_embedding)

        # 固有特徴同士の分類
        # discriminator_output_text = self.discriminator(private_text)
        # discriminator_output_audio = self.discriminator(private_audio)

        # 再構成
        recon_text = self.recon_text_linear(torch.cat([common_text, private_text], dim=-1))
        recon_audio = self.recon_audio_linear(torch.cat([common_audio, private_audio], dim=-1))

        return text_embedding, audio_embedding, common_text, common_audio, private_text, private_audio, recon_text, recon_audio
    
    
    def encode_text(self, text_x: torch.Tensor, text_attn_mask: torch.Tensor):
        text_embedding = self.text_encoder(text_x, attention_mask=text_attn_mask).last_hidden_state[:,0,:]
        common_text = self.common_text_linear(text_embedding)
        return common_text
    

    def encode_audio(self, audio_x: torch.Tensor, audio_attn_mask: torch.Tensor):
        audio_embedding = self.audio_encoder(audio_x, attention_mask=audio_attn_mask).last_hidden_state.mean(dim=1)
        common_audio = self.common_audio_linear(audio_embedding)
        return common_audio