model_type="clap"
training_dataset="audiocaps"

python -u test.py \ 
        --model_type ${model_type} \
        --dataset esc50 \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/${model_type}_${training_dataset}/epoch20.pth" \
        2>&1 | tee "logs/test/${model_type}_${training_dataset}/${dataset}.log"