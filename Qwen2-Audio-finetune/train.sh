#!/bin/bash
# ==============================
# 2-GPU 分布式训练启动脚本
# ==============================

LOCAL_DIR=/root/autodl-tmp/Qwen2-Audio-finetune
cd "$LOCAL_DIR" || exit 1

MODEL_PATH=/root/autodl-tmp/Qwen/Qwen2-Audio-7B-Instruct
TRAIN_DATA_PATH=/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/train
EVAL_DATA_PATH=/root/autodl-tmp/Qwen2-Audio-finetune/data/daic/val

TRAIN_STRATEGY=ddp
DEVICE_TYPE=cuda

# 数据加载参数
num_workers=1
prefetch_factor=2

if [[ $TRAIN_STRATEGY == "ddp" ]]; then
    # export CUDA_VISIBLE_DEVICES=0,1

    torchrun \
        --nnodes=1 \
        --nproc_per_node=1 \
        --standalone \
        main.py \
        ++train.train_strategy=$TRAIN_STRATEGY \
        ++env.device_type=$DEVICE_TYPE \
        ++env.model_path=$MODEL_PATH \
        ++data.train_data_path=$TRAIN_DATA_PATH \
        ++data.eval_data_path=$EVAL_DATA_PATH \
        ++data.num_workers=$num_workers \
        ++data.prefetch_factor=$prefetch_factor

else
    export DEEPSPEED_CONFIG=./config/deepspeed.json
    deepspeed \
        --num_nodes=1 \
        --num_gpus=1 \
        main.py \
        ++train.train_strategy=$TRAIN_STRATEGY \
        ++train.deepspeed_config=$DEEPSPEED_CONFIG \
        ++env.device_type=$DEVICE_TYPE \
        ++env.model_path=$MODEL_PATH \
        ++data.train_data_path=$TRAIN_DATA_PATH \
        ++data.eval_data_path=$EVAL_DATA_PATH
fi
