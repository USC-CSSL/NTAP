#!/bin/bash

PYTH_PATH="$(which python)"
echo "Using python located in: ${PYTH_PATH}"

WORKING_DIR=/home/brendan/neural_profiles_datadir
PROJ_NAME="MFQ-facebook"

# Directory structure:
# params/
# ----data/
# ----features/
# ----prediction/
# ----scoring/
# ----neural/

python get_params.py 
DATA_PARAMS="./params/data/default.json"
FEAT_PARAMS="./params/features/default.json"
BASE_PARAMS="./params/baselines/default.json"

echo "Processing data for project $PROJ_NAME"
./prep_data.sh $WORKING_DIR $PROJ_NAME $DATA_PARAMS

echo "Generating features"
./gen_features.sh $WORKING_DIR $PROJ_NAME $FEAT_PARAMS

echo "Building and Training baseline methods using generated/supplied features"
./train_baselines.sh $WORKING_DIR $PROJ_NAME $BASE_PARAMS
