#!/bin/bash

if [ -z "$WORKSPACE" ]; then
    echo "Must set \$WORKSPACE bash variable to path/to/work/directory, which contains source datafile"
    exit 1
fi

mkdir -p ${WORKSPACE}
export data=${WORKSPACE}/data.pkl
export features=${WORKSPACE}/features.pkl
export model=${WORKSPACE}/model.pkl
export predictions=${WORKSPACE}/predictions.pkl
export topfeatures=${WORKSPACE}/topfeatures.pkl
export random_seed=145

#  Data for the pipeline
if [ -z "$RESOURCES" ]; then
    echo "Must set \$RESOURCES bash variable to pipeline data; see README"
    exit 1
fi

export GLOVE_PATH="${RESOURCES}/WordVectors/glove/glove.6B.300d.txt";
export WORD2VEC_PATH="${RESOURCES}/WordVectors/word2vec/GoogleNews-vectors-negative300.bin";
export FASTTEXT_PATH="${RESOURCES}/WordVectors/fasttext/wiki.en.bin";
export INFERSENT_PATH="${RESOURCES}/PretrainedModels/infersent/infersent.allnli.pickle";
export DICTIONARIES="${RESOURCES}/Dictionaries/";

