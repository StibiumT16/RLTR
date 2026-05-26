gpuid=0
algorithm=ce

datasets=("mslr" "istella" "yahoo")
lrs=(0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01)

for dataset in "${datasets[@]}"; do
    for lr in "${lrs[@]}"; do
        CUDA_VISIBLE_DEVICES=$gpuid python src/main.py -o train \
        --input_feed deterministic_online_label_input \
        --config_path config/${algorithm}.yaml \
        --data_path dataset/$dataset \
        --log_path log/${dataset}/${algorithm}_${lr}/ \
        --model_save_path model/${dataset}/${algorithm}_${lr}.pt \
        -l $lr -s 20000

        CUDA_VISIBLE_DEVICES=$gpuid python src/main.py -o test \
        --config_path config/${algorithm}.yaml \
        --data_path dataset/$dataset \
        --model_save_path model/${dataset}/${algorithm}_${lr}.pt \
        --output_path output/${dataset}/${algorithm}_${lr}.csv
    done
done
