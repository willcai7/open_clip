#!/bin/bash
cd src

batch_size=128

# CLIP Loss 
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 30 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --train-data="../data/cc3m-wds/cc3m-train-{0000..0575}.tar" \
    --imagenet-val="../data/imagenet-1k/val/" \
    --train-num-samples 100000 \
    --dataset-resampled \
    --seed 42 \
    --warmup 1000 \
    --batch-size $batch_size \
    --accum-freq 1 \
    --precision amp \
    --lr=1e-3 \
    --wd=0.01 \
    --log-every-n-steps 5 \
    --epochs=32 \
    --workers=8 \
    --model RN50 

# wd_list=(1 0.3 0.03 0.01)
lr_list=(1e-2 3e-3 1e-3 6e-4 3e-4 1e-4 3e-5)

# Chi Loss 
for lr in ${lr_list[@]}; do
    echo "Running with lr=$lr"
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 30 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --train-data="../data/cc3m-wds/cc3m-train-{0000..0575}.tar" \
    --imagenet-val="../data/imagenet-1k/val/" \
    --train-num-samples 100000 \
    --dataset-resampled \
    --seed 42 \
    --warmup 1000 \
    --batch-size $batch_size \
    --accum-freq 1 \
    --precision amp \
    --lr=$lr \
    --wd=0.01 \
    --log-every-n-steps 5 \
    --epochs=32 \
    --workers=8 \
    --model RN50 \
    --chi 
done 

# Spec Loss 
for lr in ${lr_list[@]}; do
    echo "Running with lr=$lr"
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 30 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --train-data="../data/cc3m-wds/cc3m-train-{0000..0575}.tar" \
    --imagenet-val="../data/imagenet-1k/val/" \
    --train-num-samples 100000 \
    --dataset-resampled \
    --seed 42 \
    --warmup 1000 \
    --batch-size $batch_size \
    --accum-freq 1 \
    --precision amp \
    --lr=$lr \
    --wd=0.01 \
    --log-every-n-steps 5 \
    --epochs=32 \
    --workers=8 \
    --model RN50 \
    --spec 
done

# Chi zero Loss
for lr in ${lr_list[@]}; do
    echo "Running with lr=$lr"
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 30 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --train-data="../data/cc3m-wds/cc3m-train-{0000..0575}.tar" \
    --imagenet-val="../data/imagenet-1k/val/" \
    --train-num-samples 100000 \
    --dataset-resampled \
    --seed 42 \
    --warmup 1000 \
    --batch-size $batch_size \
    --accum-freq 1 \
    --precision amp \
    --lr=$lr \
    --wd=0.01 \
    --log-every-n-steps 5 \
    --epochs=32 \
    --workers=8 \
    --model RN50 \
    --chizero 
done

# SigLip Loss 
for lr in ${lr_list[@]}; do
    echo "Running with lr=$lr"
torchrun --nproc_per_node 4 -m open_clip_train.main \
    --save-frequency 30 \
    --zeroshot-frequency 1 \
    --report-to "wandb" \
    --train-data="../data/cc3m-wds/cc3m-train-{0000..0575}.tar" \
    --imagenet-val="../data/imagenet-1k/val/" \
    --train-num-samples 100000 \
    --dataset-resampled \
    --seed 42 \
    --warmup 1000 \
    --batch-size $batch_size \
    --accum-freq 1 \
    --precision amp \
    --lr=$lr \
    --wd=0.01 \
    --log-every-n-steps 5 \
    --epochs=32 \
    --workers=8 \
    --model RN50 \
    --siglip 
done
