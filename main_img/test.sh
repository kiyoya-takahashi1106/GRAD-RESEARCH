model_type="method"   # help: "clip", "method", "method2",  "ablation"
training_dataset="coco"
datasets=("oxford_pet" "caltech101")
hidden_dim=768   # 768 or 1024
zs_type="common"   # "common" or "private" or "cp" or "pc" or "concat"

mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

seeds=(41 42 43)

for dataset in "${datasets[@]}"; do
     mkdir -p logs/test_${hidden_dim}/${model_type}_${training_dataset}

    for seed in "${seeds[@]}"; do
        python -u test.py \
            --model_type ${model_type} \
            --dataset ${dataset} \
            --hidden_dim ${hidden_dim} \
            --zs_type ${zs_type} \
            --dropout_rate 0.1 \
            --saved_model_path "./saved_models/${hidden_dim}/${model_type}_${training_dataset}/best${seed}.pth" \
            2>&1 | { [ "$zs_type" = "common" ] && tee "logs/test_${hidden_dim}/${model_type}_${training_dataset}/${dataset}_${seed}.log" || cat; }
    done
done