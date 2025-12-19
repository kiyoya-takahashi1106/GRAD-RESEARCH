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

    def compute_loss(self, text_embedding, audio_embedding, common_text, common_audio, private_text, private_audio, recon_text, recon_audio):
        """
        Contrastive loss
        L2 loss
        Diff Loss *3
        Reconstruction loss *2
        """
        contractive_loss = self.compute_contrastive_loss(common_text, common_audio)
        sim_loss = self.compute_sim_loss(common_text, common_audio)
        c2p_text_loss = self.compute_cka_loss(common_text, private_text)
        c2p_audio_loss = self.compute_cka_loss(common_audio, private_audio)
        p2p_loss = self.compute_cka_loss(private_text, private_audio)
        reconstruction_text_loss = self.compute_reconstruction_loss(text_embedding, recon_text)
        reconstruction_audio_loss = self.compute_reconstruction_loss(audio_embedding, recon_audio)
        return contractive_loss, sim_loss, c2p_text_loss, c2p_audio_loss, p2p_loss, reconstruction_text_loss, reconstruction_audio_loss
    
    
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
        # common_text_norm = F.normalize(common_text, p=2, dim=-1)
        # common_audio_norm = F.normalize(common_audio, p=2, dim=-1)
        # sim = torch.sum(common_text_norm * common_audio_norm, dim=-1)
        # sim = sim.mean()
        # sim_loss = 1.0 - sim
        # return sim_loss

        # CKAに基づく類似度損失を計算する
        X = common_text - common_text.mean(dim=0, keepdim=True)
        Y = common_audio - common_audio.mean(dim=0, keepdim=True)
        XT_Y = X.T @ Y
        XTX = X.T @ X
        YTY = Y.T @ Y
        hsic = (XT_Y ** 2).sum()
        eps = 1e-8
        norm_x = torch.sqrt((XTX ** 2).sum() + eps)
        norm_y = torch.sqrt((YTY ** 2).sum() + eps)
        cka = hsic / (norm_x * norm_y + eps)
        sim_loss = 1.0 - cka
        return sim_loss


    def compute_reverse_contrastive_loss(self, private_text, private_audio):
        """
        固有特徴に対して、逆対照学習を行う損失を計算する
        """
        batch_size = private_text.size(0)
        temperature = 0.1

        # 正規化
        private_text_norm = F.normalize(private_text, p=2, dim=1)
        private_audio_norm = F.normalize(private_audio, p=2, dim=1)

        # 類似度計算
        logits = torch.matmul(private_text_norm, private_audio_norm.T) / temperature
        
        labels = torch.arange(batch_size).to(private_text.device)

        loss_text_to_audio = F.cross_entropy(logits, labels)
        loss_audio_to_text = F.cross_entropy(logits.T, labels)

        loss = (loss_text_to_audio + loss_audio_to_text) / 2
        return loss
    

    def compute_cka_loss(self, feature1, feature2):
        """
        CKAに基づく差異損失を計算する
        1 に近い → 空間として非常に似ている
        0 に近い → 全然違う構造
        """
        # サンプル方向で平均を引いてセンタリング
        X = feature1 - feature1.mean(dim=0, keepdim=True)   # (N, D_text)
        Y = feature2 - feature2.mean(dim=0, keepdim=True) # (N, D_audio)

        # サンプル間のにてる度合い
        XT_Y = X.T @ Y       # (D_text, D_audio)
        XTX  = X.T @ X       # (D_text, D_text)
        YTY  = Y.T @ Y       # (D_audio, D_audio)

        # Frobenius ノルムの二乗（要素二乗和）
        hsic = (XT_Y ** 2).sum()

        eps = 1e-8
        norm_x = torch.sqrt((XTX ** 2).sum() + eps)
        norm_y = torch.sqrt((YTY ** 2).sum() + eps)

        cka = hsic / (norm_x * norm_y + eps)
        return cka


    # def compute_reverse_contrastive_loss(self, private_text, private_audio):
    #     """
    #     固有特徴に対して、soft label（一様分布）を使った逆対照学習。
    #     各行の softmax を 1/B に近づけ、特定ペアだけが大きくなるのを防ぐ。
    #     """
    #     batch_size = private_text.size(0)
    #     temperature = 0.1

    #     # 正規化
    #     private_text_norm = F.normalize(private_text, p=2, dim=1)      # (B, D)
    #     private_audio_norm = F.normalize(private_audio, p=2, dim=1)     # (B, D)

    #     # 類似度計算
    #     logits = torch.matmul(private_text_norm, private_audio_norm.T) / temperature

    #     # 一様分布 p = [1/B, 1/B, ..., 1/B]
    #     uniform_targets = torch.full((batch_size, batch_size), 1.0/batch_size, device=private_text.device)

    #     # text → audio: row-wise softmax
    #     log_probs_text_to_audio = F.log_softmax(logits, dim=1)
    #     loss_text_to_audio = -(uniform_targets * log_probs_text_to_audio).sum(dim=1).mean()

    #     # audio → text: column-wise softmax
    #     log_probs_audio_to_text = F.log_softmax(logits.T, dim=1)
    #     loss_audio_to_text = -(uniform_targets * log_probs_audio_to_text).sum(dim=1).mean()

    #     loss = (loss_text_to_audio + loss_audio_to_text) / 2
    #     return loss


    def compute_reconstruction_loss(self, embedding, recon):
        """
        再構成損失を計算する
        """
        loss = F.mse_loss(embedding, recon)
        return loss