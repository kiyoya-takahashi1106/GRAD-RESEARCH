# ===== TRAINING =====
model_type="clap"   # help: "clap", "method" 
dataset="mix"   # help: "audiocaps", "fsd50k", "clotho", "macs", "mix"
hidden_dim=768   # 768 or 1024

mkdir -p logs/train_${hidden_dim}/${model_type}_${dataset}

python -u train.py \
    --model_type ${model_type} \
    --seed 42 \
    --dataset ${dataset} \
    --lr 1e-3 \
    --epochs 200 \
    --batch_size 256 \
    --hidden_dim ${hidden_dim} \
    --dropout_rate 0.1 \
    --hp_contrastive 0.2 \
    --hp_sim 0.2 \
    --hp_cp_diff 1.0 \
    --hp_pp_diff 1.0 \
    --hp_recon 2.0 \
    2>&1 | tee "logs/train_${hidden_dim}/${model_type}_${dataset}.log"



# ===== TESTING =====
training_dataset="${dataset}"
dataset="esc50"

mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --hidden_dim ${hidden_dim} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/${hidden_dim}/${model_type}_${training_dataset}/best.pth" \
        2>&1 | tee "logs/test_${hidden_dim}/${model_type}_${training_dataset}/${dataset}.log"