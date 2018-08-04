#!/bin/bash

WORKING_DIR=$1
DATA_TYPE=$2
PARAMS=$3

SOURCE="$WORKING_DIR/data/${DATA_TYPE}/source.csv"
DEST="$WORKING_DIR/data/${DATA_TYPE}/dataframe.pkl"
SCRIPT="./data/${DATA_TYPE}/prep.py"

if [ -e $DEST ]
then
    echo "File exists; to regenerate, delete $DEST and rerun"
else
    python $SCRIPT $SOURCE $DEST $PARAMS
fi
