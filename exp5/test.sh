python -u test.py \
        --dataset esc50 \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/mix/best.pth" \
        2>&1 | tee "logs/test/mix/esc50.log"