#!/bin/bash

PREP_SCRIPT="./data/${PROJ_NAME}/prep.py"
PROCESS_SCRIPT="./preprocess/preprocess.py"

python $PREP_SCRIPT 
python $PROCESS_SCRIPT
