#!/bin/bash

RECORD=mixb
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/testmixformer.yaml

WEIGHTS=/home/niyunfei/workspace/wuRenji/wuRenji/log/mixformer_bone_ckpt-0-0.pt


BATCH_SIZE=256

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
