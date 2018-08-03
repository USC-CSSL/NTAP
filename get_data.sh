#!/bin/bash

PYTH_PATH="$(which python)"
echo "Using python located in: ${PYTH_PATH}"


DATA_PATH=$1
DATA_TYPE="MFQ-facebook"
DEST="./data/${DATA_TYPE}/"

python clean.py "${DEST}/
