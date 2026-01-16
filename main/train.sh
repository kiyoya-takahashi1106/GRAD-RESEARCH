# ===== TRAINING =====
model_type="method2"   # help: "clap", "method", "method2"
dataset="mix"   # help: "audiocaps", "fsd50k", "clotho",  "mix"
batch_size=256   # help: 32, 64, 128, 256, 512, 768
hidden_dim=768   # 768 or 1024

seeds=(41 42 43)

mkdir -p logs/train_${hidden_dim}/${model_type}_${dataset}

for seed in "${seeds[@]}"; do
    python -u train.py \
        --model_type ${model_type} \
        --seed ${seed} \
        --dataset ${dataset} \
        --lr 1e-3 \
        --epochs 40 \
        --batch_size ${batch_size} \
        --hidden_dim ${hidden_dim} \
        --dropout_rate 0.1 \
        --sim_loss_type "cka" \
        --hp_contrastive 0.2 \
        --hp_sim 0.2 \
        --hp_cp_diff 0.0 \
        --hp_pp_diff 1.0 \
        --hp_recon 2.0 \
        2>&1 | tee "logs/train_${hidden_dim}/${model_type}_${dataset}/${seed}.log"
done



# ===== TESTING =====
training_dataset="${dataset}"
dataset="esc50"   # "esc50" or "us8k" or "beijing_opera" or "vocal_sound" or "tut2017" or "mri_stroke"

mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

for seed in "${seeds[@]}"; do
    python -u test.py \
        --model_type ${model_type} \
        --dataset ${dataset} \
        --hidden_dim ${hidden_dim} \
        --zs_type "common" \
        --dropout_rate 0.1 \
        --saved_model_path "./saved_models/${hidden_dim}/${model_type}_${training_dataset}/best${seed}.pth" \
        2>&1 | tee "logs/test_${hidden_dim}/${model_type}_${training_dataset}/${dataset}_${seed}.log"
done