python -u test.py \
        --dataset esc50 \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/audiocaps/epoch10.pth" \
        2>&1 | tee "logs/test.log"