#!/bin/bash
RECORD=fft_j
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD


CONFIG=./config/train_fft.yaml

START_EPOCH=60
EPOCH_NUM=60
BATCH_SIZE=64
WARM_UP=5
SEED=777

WEIGHTS=/home/niyunfei/workspace/wuRenji/wuRenji/wuRenji/runs/fft_j-40-10496.pt


python3 main.py --weights $WEIGHTS --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --warm_up_epoch $WARM_UP --only_train_epoch $EPOCH_NUM --seed $SEED

