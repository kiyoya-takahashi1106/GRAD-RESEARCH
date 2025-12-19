# ===== TRAINING =====
dataset="mix"   # help: "mix"
mkdir -p logs/train

python -u train.py \
        --seed 42 \
        --dataset ${dataset} \
        --lr 1e-3 \
        --epochs 50 \
        --start_sim_epoch 5 \
        --batch_size 256 \
        --dropout_rate 0.1 \
        --hp_contrastive 0.2 \
        --hp_sim 0.2 \
        --hp_cp_diff 1.0 \
        --hp_pp_diff 1.0 \
        --hp_recon 2.0 \
        2>&1 | tee "logs/train/${dataset}.log"



# ===== TESTING =====
training_dataset="${dataset}"
dataset="esc50"

mkdir -p logs/test/${training_dataset}

python -u test.py \
        --dataset ${dataset} \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/${training_dataset}/best.pth" \
        2>&1 | tee "logs/test/${training_dataset}/${dataset}.log"