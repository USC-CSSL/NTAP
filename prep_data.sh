#!/bin/bash

PREP_SCRIPT="./data/${PROJ_NAME}/prep.py"
PROCESS_SCRIPT="./preprocess/preprocess.py"
PARTITION_SCRIPT="./data/${PROJ_NAME}/make_datasets.py"

python $PREP_SCRIPT 
python $PROCESS_SCRIPT
python $PARTITION_SCRIPT
