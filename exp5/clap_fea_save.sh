seed=42
datasets=("fsd50k")
batch_size=200

for dataset in "${datasets[@]}"; do
    python -u clap_fea_save.py \
        --seed ${seed} \
        --dataset ${dataset} \
        --batch_size ${batch_size}
done