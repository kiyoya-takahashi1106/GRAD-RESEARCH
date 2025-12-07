model_type="clap"
dataset="audiocaps"

mkdir -p logs/${model_type}_${dataset}

python -u train.py \
    --model_type ${model_type} \
    --seed 42 \
    --dataset ${dataset} \
    --lr 1e-4 \
    --epochs 40 \
    --batch_size 50 \
    --dropout_rate 0.1 \
    --hp_contrastive 0.2 \
    --hp_sim 1.0 \
    --hp_discrim 1.0 \
    --hp_recon 1.0 \
    2>&1 | tee "logs/${model_type}_${dataset}/train.log"
