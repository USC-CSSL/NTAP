#!/bin/bash

export PYTHONPATH="$(pwd):${PYTHONPATH}"

export WORKING_DIR="/home/brendan/neural_profiles_datadir/"
export PROJ_NAME="MFQ-facebook"
export INSTANCE_NAME="classify_averages"

python get_params.py

. set-vars.sh $WORKING_DIR $PROJ_NAME $INSTANCE_NAME

#./prep_data.sh

#./gen_features.sh

./train_baselines.sh

#./train_neural.sh

#./evaluate.sh
