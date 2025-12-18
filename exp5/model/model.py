"""
CLAPの出力に対して、共通-固有分離を用いて共通特徴を抽出するモデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, dropout_rate: float, hidden_dim: int = 1024, saved_model_path: str = None):
        super(Model, self).__init__()

        # 共通-固有分離用のMLP×4
        self.common_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_text_linear = nn.Linear(hidden_dim, hidden_dim)
        self.common_audio_linear = nn.Linear(hidden_dim, hidden_dim)
        self.private_audio_linear = nn.Linear(hidden_dim, hidden_dim)

        # 固有同士の分類をするdiscriminator
        self.discriminator = nn.Linear(hidden_dim, 1)

        # 再構成用のデコーダ
        self.recon_text_linear = nn.Linear(hidden_dim*2, hidden_dim)
        self.recon_audio_linear = nn.Linear(hidden_dim*2, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        # val用に学習済みモデルをロード
        if (saved_model_path is not None):
            self.load_state_dict(torch.load(saved_model_path))
            print(f"Loaded trained model from {saved_model_path}")


    def forward(self, text_embedding: torch.Tensor, audio_embedding: torch.Tensor):
        # 共通-固有分離
        common_text = self.common_text_linear(text_embedding)
        private_text = self.private_text_linear(text_embedding)
        common_audio = self.common_audio_linear(audio_embedding)
        private_audio = self.private_audio_linear(audio_embedding)

        # 固有特徴同士の分類
        discriminator_output_text = self.discriminator(private_text)
        discriminator_output_audio = self.discriminator(private_audio)

        # 再構成
        recon_text = self.recon_text_linear(torch.cat([common_text, private_text], dim=-1))
        recon_audio = self.recon_audio_linear(torch.cat([common_audio, private_audio], dim=-1))

        return common_text, common_audio, discriminator_output_text, discriminator_output_audio, recon_text, recon_audio
    

    def extract_common_text(self, text_embedding: torch.Tensor):
        common_text = self.common_text_linear(text_embedding)
        return common_text
    

    def extract_common_audio(self, audio_embedding: torch.Tensor):
        common_audio = self.common_audio_linear(audio_embedding)
        return common_audio