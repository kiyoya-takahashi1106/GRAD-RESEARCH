seed=42
datasets=("macs")
batch_size=120
hidden_dim=768   # 768 or 1024

for dataset in "${datasets[@]}"; do
    python -u fea_save.py \
        --seed ${seed} \
        --dataset ${dataset} \
        --batch_size ${batch_size} \
        --hidden_dim ${hidden_dim}
done
