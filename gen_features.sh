#!/bin/bash

WORKING_DIR=$1
PROJ_NAME=$2
PARAMS=$3

SOURCE=${WORKING_DIR}/data/${PROJ_NAME}/dataframe.pkl
DEST_DIR=${WORKING_DIR}/features/${PROJ_NAME}
GEN_SCRIPT='./feature-gen/make_features.py'


mkdir -p ${DEST_DIR}

python -c "import pandas as pd; print(pd.read_pickle('${SOURCE}'))"

echo "Generating features from text; saving to $DEST_DIR"
python ${GEN_SCRIPT}

