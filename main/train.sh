python -u train.py \
        --model_type clap \
        --seed 42 \
        --dataset mix \
        --lr 1e-4 \
        --epochs 200 \
        --batch_size 2000 \
        --dropout_rate 0.1 \
        --hp_contrastive 0.2 \
        --hp_sim 1.0 \
        --hp_discrim 1.0 \
        --hp_recon 1.0 \
        2>&1 | tee "logs/train.log"