seed=42
dataset="fsd50k"
batch_size=100

python -u fea_save.py \
        --seed ${seed} \
        --dataset ${dataset} \
        --batch_size ${batch_size} \