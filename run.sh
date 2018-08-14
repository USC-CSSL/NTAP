#!/bin/bash


PYTH_PATH="python3"
echo "Using python located in: ${PYTH_PATH}"

alias python= PYTH_PATH

export WORKING_DIR=/home/aida/neural_profiles_datadir
export PROJ_NAME="MFQ-facebook"
export INSTANCE_NAME="test_run"

python get_params.py

. set-vars.sh $WORKING_DIR $PROJ_NAME $INSTANCE_NAME

echo "Processing data for project $PROJ_NAME"
#./prep_data.sh

echo "Generating features"
#./gen_features.sh

echo "Building and Training baseline methods using generated/supplied features"
#./train_baselines.sh

echo "Building and Training neural network"
./train_neural.sh

#./evaluate.sh
