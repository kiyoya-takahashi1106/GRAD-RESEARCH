python -u val.py \
        --dataset esc50 \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/audiocaps/epoch181.pth" \
        2>&1 | tee "logs/val.log"