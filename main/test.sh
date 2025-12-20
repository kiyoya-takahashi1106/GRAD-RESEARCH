model_type="method"
training_dataset="mix"
dataset="esc50"
hidden_dim=1024   # 768 or 1024

mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --hidden_dim ${hidden_dim} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/train/${model_type}_${training_dataset}/epoch36.pth" \
        2>&1 | tee "logs/test_${hidden_dim}/${model_type}_${training_dataset}/${dataset}.log"