#!/bin/bash
RECORD=ctrgcn_gcl_bone_motion
CONFIG=./config/train_ctrgcn_gcl_bm.yaml


WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD
START_EPOCH=0
EPOCH_NUM=60
BATCH_SIZE=56
WARM_UP=5
SEED=777

# WEIGHT=/home/niyunfei/workspace/wuRenji/wuRenji/wuRenji/runs/ctrgcn_gcl_joint-49-14650.pt

python3 main_gcl.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED
