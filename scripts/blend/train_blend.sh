#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="mvoco-mvsnet+"

LOG_DIR="./checkpoints/blend/"$THISNAME 
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=0 python3 train.py ${@} \
    --which_dataset="blendedmvs" --which_module="mvoco_plus" --epochs=16 --logdir=$LOG_DIR \
    --trainpath=$BLENDEDMVS_ROOT --testpath=$BLENDEDMVS_ROOT \
    --trainlist="datasets/lists/blendedmvs/low_res_all.txt" --testlist="datasets/lists/blendedmvs/val.txt" \
    \
    --n_views="9" --batch_size=2 --lr=0.001 --robust_train \
    --lr_scheduler="onecycle"