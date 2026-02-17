"""
main_imgで使う
データ前処理用のcollate_fn
"""

import torch


def collate_fn_img(batch, text_tokenizer, img_processor):
    # ==== Split ====
    splits = [sample[0] for sample in batch]


    # ===== Text =====
    texts = [sample[1] for sample in batch]
    tok = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = tok["input_ids"]          # (B, L_text)
    text_attn_mask = tok["attention_mask"]     # (B, L_text)


    # ===== Img =====
    # COCODataset からは PIL.Image.Image が来る想定
    raw_imgs = [sample[2] for sample in batch]

    # ViTImageProcessor: (B, 3, 224, 224) の pixel_values を返す
    img_out = img_processor(
        images=raw_imgs,
        return_tensors="pt",
    )
    img_tensor = img_out["pixel_values"]       # (B, 3, 224, 224)

    # ViT は通常 attention_mask を使わないので None でOK
    img_attn_mask = None

    return splits, text_input_ids, text_attn_mask, img_tensor, img_attn_mask
