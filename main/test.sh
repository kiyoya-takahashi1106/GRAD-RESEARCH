model_type="method"
training_dataset="mix"
dataset="beijing_opera"   # "esc50" or "us8k" or "beijing_opera"
hidden_dim=768   # 768 or 1024
zs_type="common"   # "common" or "private" or "add" or "concat"

mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

seeds=(41 42 43)

for seed in "${seeds[@]}"; do
    python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --hidden_dim ${hidden_dim} \
        --zs_type ${zs_type} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/${hidden_dim}/${model_type}_${training_dataset}/best${seed}.pth" \
        2>&1 | tee "logs/test_${hidden_dim}/${model_type}_${training_dataset}/${dataset}_${seed}.log"
done