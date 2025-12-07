"""
mainで使う
データ前処理用のcollate_fn
"""


def collate_fn(batch, text_tokenizer, audio_processor):
    texts = [sample[0] for sample in batch]
    audios = [sample[1] for sample in batch]

    tokenizer_output = text_tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids=tokenizer_output["input_ids"]
    text_attn_mask=tokenizer_output["attention_mask"]

    processor_output = audio_processor(
        audios,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
    )
    audio_input_values = processor_output["input_values"]
    audio_attn_mask = processor_output["attention_mask"]

    return text_input_ids, text_attn_mask, audio_input_values, audio_attn_mask