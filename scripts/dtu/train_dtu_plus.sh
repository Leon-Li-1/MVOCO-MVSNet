#!/usr/bin/env bash
source scripts/data_path.sh

THISNAME="mvoco-mvsnet+"

LOG_DIR="./checkpoints/dtu/"$THISNAME 
if [ ! -d $LOG_DIR ]; then
    mkdir -p $LOG_DIR
fi

CUDA_VISIBLE_DEVICES=0 python3 train.py ${@} \
    --which_dataset="dtu" --which_module="mvoco_plus" --epochs=16 --logdir=$LOG_DIR \
    --trainpath=$DTU_TRAIN_ROOT --testpath=$DTU_TRAIN_ROOT \
    --trainlist="datasets/lists/dtu/train.txt" --testlist="datasets/lists/dtu/test.txt" \
    \
    --data_scale="mid" --n_views="5" --batch_size=4 --lr=0.002 --robust_train \
    --lrepochs="6,10,12,14:2"
