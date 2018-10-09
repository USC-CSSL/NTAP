#!/bin/bash

# Data-processing Paths: Project-specific
export RAW_PATH="$WORKING_DIR/data/${PROJ_NAME}/source.csv";
export SOURCE_DIR="$WORKING_DIR/data/${PROJ_NAME}/${INSTANCE_NAME}";
mkdir -p "$SOURCE_DIR"
export SOURCE_PATH="${SOURCE_DIR}/full_df.pkl";

export FEAT_DIR=${WORKING_DIR}/features/${PROJ_NAME}/${INSTANCE_NAME};
export PRED_DIR=${WORKING_DIR}/predictions/${PROJ_NAME}/${INSTANCE_NAME}/;
mkdir -p "$PRED_DIR"
mkdir -p "$FEAT_DIR"

# External data source paths
export GLOVE_PATH="${WORKING_DIR}/word_embeddings/GloVe/glove.6B.300d.txt";
export WORD2VEC_PATH="${WORKING_DIR}/word_embeddings/skipgram/GoogleNews-vectors-negative300.bin";
export FASTTEXT_PATH="${WORKING_DIR}/sent_embeddings/fasttext/wiki.en.bin";
export INFERSENT_PATH="${WORKING_DIR}/sent_embeddings/infersent/infersent.allnli.pickle";
export MFD_PATH="${WORKING_DIR}/dictionaries/moral_foundations_theory.json";

# Params paths
export PARAMS="./params/${INSTANCE_NAME}.json";
