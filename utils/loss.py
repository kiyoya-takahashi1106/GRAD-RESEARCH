import torch
import torch.nn as nn
import torch.nn.functional as F


class ClapCriterion:
    def __init__(self):
        pass

    def compute_loss(self, text_embedding, audio_embedding):
        """
        CLAPのコントラスト学習損失を計算する
        """
        batch_size = text_embedding.size(0)
        temperature = 0.1

        # 正規化
        text_embedding_norm = F.normalize(text_embedding, p=2, dim=1)
        audio_embedding_norm = F.normalize(audio_embedding, p=2, dim=1)

        # 類似度計算
        logits = torch.matmul(text_embedding_norm, audio_embedding_norm.T) / temperature

        labels = torch.arange(batch_size).to(text_embedding.device)

        loss_text_to_audio = F.cross_entropy(logits, labels)
        loss_audio_to_text = F.cross_entropy(logits.T, labels)

        loss = (loss_text_to_audio + loss_audio_to_text) / 2
        return loss
    

class Criterion:
    def __init__(self):
        pass

    def compute_loss(self, text_embedding, audio_embedding, common_text, common_audio, discriminator_output_text, discriminator_output_audio, recon_text, recon_audio):
        """
        Contrastive loss
        L2 loss
        Discriminator loss *2
        Reconstruction loss *2
        """
        contractive_loss = self.compute_contrastive_loss(common_text, common_audio)
        sim_loss = self.compute_sim_loss(common_text, common_audio)
        discriminator_text_loss = self.compute_discriminator_loss(discriminator_output_text, "text")
        discriminator_audio_loss = self.compute_discriminator_loss(discriminator_output_audio, "audio")
        reconstruction_text_loss = self.compute_reconstruction_loss(text_embedding, recon_text)
        reconstruction_audio_loss = self.compute_reconstruction_loss(audio_embedding, recon_audio)
        return contractive_loss, sim_loss, discriminator_text_loss, discriminator_audio_loss, reconstruction_text_loss, reconstruction_audio_loss
    
    def compute_contrastive_loss(self, common_text, common_audio):
        """
        共通特徴に対して、コントラスト学習を行う
        """
        batch_size = common_text.size(0)
        temperature = 0.1

        # 正規化
        common_text_norm = F.normalize(common_text, p=2, dim=1)
        common_audio_norm = F.normalize(common_audio, p=2, dim=1)

        # 類似度計算
        logits = torch.matmul(common_text_norm, common_audio_norm.T) / temperature

        labels = torch.arange(batch_size).to(common_text.device)

        loss_text_to_audio = F.cross_entropy(logits, labels)
        loss_audio_to_text = F.cross_entropy(logits.T, labels)

        loss = (loss_text_to_audio + loss_audio_to_text) / 2
        return loss

    def compute_sim_loss(self, common_text, common_audio):
        """
        共通特徴に対して、内積を計算する
        """
        common_text_norm = F.normalize(common_text, p=2, dim=-1)
        common_audio_norm = F.normalize(common_audio, p=2, dim=-1)
        sim = torch.sum(common_text_norm * common_audio_norm, dim=-1)
        sim = sim.mean()
        sim_loss = 1.0 - sim
        return sim_loss

    def compute_discriminator_loss(self, discriminator_output, modality: str):
        """
        固有特徴に対して、discriminatorの損失を計算する
        """
        modality_label = 0 if modality == "text" else 1
        labels = torch.full((discriminator_output.size(0),), modality_label, dtype=torch.long, device=discriminator_output.device)
        loss = F.binary_cross_entropy_with_logits(discriminator_output.squeeze(), labels.float())
        return loss

    def compute_reconstruction_loss(self, embedding, recon):
        """
        再構成損失を計算する
        """
        loss = F.mse_loss(embedding, recon)
        return loss