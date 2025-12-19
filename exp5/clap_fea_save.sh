seed=42
datasets=("macs")
batch_size=200

for dataset in "${datasets[@]}"; do
    python -u clap_fea_save.py \
        --seed ${seed} \
        --dataset ${dataset} \
        --batch_size ${batch_size}
done