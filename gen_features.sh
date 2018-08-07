#!/bin/bash

GEN_SCRIPT='./feature-gen/make_features.py'

echo "Generating features from text; saving to $FEAT_PATH"
python ${GEN_SCRIPT} 

