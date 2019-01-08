#!/bin/bash

export RESOURCES=${HOME}/PipelineResources

export WORKSPACE=$1
export sourcedata=$2
. set-vars.sh

#python process.py --input $sourcedata --output $data 
#python features.py --input $data --output $features
#python predict.py --data $data --features $features \
#                  --results $predictions --topfeatures $topfeatures

python scoring.py --predictions $predictions
