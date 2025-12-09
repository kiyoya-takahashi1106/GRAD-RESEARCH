model_type="clap"
training_dataset="audiocaps"
dataset="esc50"

mkdir -p logs/test/${model_type}_${training_dataset}

python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/train/${model_type}_${training_dataset}/epoch0.pth" \
        2>&1 | tee "logs/test/${model_type}_${training_dataset}/${dataset}.log"