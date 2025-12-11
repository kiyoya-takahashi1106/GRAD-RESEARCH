# ===== TRAINING =====
model_type="method"
dataset="audiocaps"

mkdir -p logs/train/${model_type}_${dataset}

python -u train.py \
    --model_type ${model_type} \
    --seed 42 \
    --dataset ${dataset} \
    --lr 1e-3 \
    --epochs 100 \
    --batch_size 1000 \
    --dropout_rate 0.1 \
    --hp_contrastive 0.2 \
    --hp_sim 0 \
    --hp_diff 1.0 \
    --hp_recon 2.0 \
    2>&1 | tee "logs/train/${model_type}_${dataset}/train.log"



# ===== TESTING =====
training_dataset="${dataset}"
dataset="esc50"

mkdir -p logs/test/${model_type}_${training_dataset}

python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/train/${model_type}_${training_dataset}/best.pth" \
        2>&1 | tee "logs/test/${model_type}_${training_dataset}/${dataset}.log"