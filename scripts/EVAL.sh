#!/bin/bash

RECORD=stt_b
WORKDIR=work_dir/$RECORD
MODELNAME=runs/$RECORD

CONFIG=./config/testctr.yaml

WEIGHTS=./


BATCH_SIZE=64

python3 main.py  --config $CONFIG --work-dir $WORKDIR -model_saved_name $MODELNAME --device 0 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --weights $WEIGHTS
