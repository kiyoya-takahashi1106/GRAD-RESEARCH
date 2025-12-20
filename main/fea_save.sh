seed=42
datasets=("audiocaps" "fsd50k" "clotho" "macs")
batch_size=80
dim_num=1024   # 768 or 1024

for dataset in "${datasets[@]}"; do
    python -u fea_save.py \
        --seed ${seed} \
        --dataset ${dataset} \
        --batch_size ${batch_size} \
        --dim_num ${dim_num}
done
