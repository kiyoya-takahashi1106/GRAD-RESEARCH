"""
mainで使う
データ前処理用のcollate_fn
"""

import torch

def random_crop_or_pad(wav: torch.Tensor, target_len: int = 16000 * 5):
    L = wav.size(0)

    # 5秒ちょうど or 5秒より長い場合
    if (L >= target_len):
        if (L > target_len):
            start = torch.randint(0, L - target_len + 1, (1,)).item()
            wav = wav[start:start + target_len]
        # L == target_len のときはそのまま
        attn = torch.ones(target_len, dtype=torch.long)
        return wav, attn

    # 5秒より短い場合 → 右側ゼロパディング
    pad_len = target_len - L
    padded = torch.cat([wav, torch.zeros(pad_len, device=wav.device)], dim=0)

    attn = torch.cat([
        torch.ones(L, dtype=torch.long, device=wav.device),
        torch.zeros(pad_len, dtype=torch.long, device=wav.device)
    ], dim=0)

    return padded, attn


def collate_fn(batch, text_tokenizer, audio_processor):
    # ==== Split ====
    splits = [sample[0] for sample in batch]

    # ===== Text =====
    texts = [sample[1] for sample in batch]

    tokenizer_output = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = tokenizer_output["input_ids"]          # (B, L_text)
    text_attn_mask = tokenizer_output["attention_mask"]     # (B, L_text)

    # ===== Audio =====
    raw_audios = [sample[2] for sample in batch]  # list of Tensor(T,)

    cropped_padded = []
    audio_masks = []

    for wav in raw_audios:
        wav_5s, attn = random_crop_or_pad(wav)
        cropped_padded.append(wav_5s)
        audio_masks.append(attn)

    # (B, T) にまとめる
    cropped_padded = torch.stack(cropped_padded, dim=0)
    audio_attn_mask = torch.stack(audio_masks, dim=0)   

    # すでに長さは揃っているので padding=False
    processor_output = audio_processor(
        cropped_padded,        
        sampling_rate=16000,
        padding=False,
        return_tensors="pt",
    )
    audio_input_values = processor_output["input_values"]
    audio_input_values = audio_input_values.squeeze(0)

    return splits, text_input_ids, text_attn_mask, audio_input_values, audio_attn_mask
