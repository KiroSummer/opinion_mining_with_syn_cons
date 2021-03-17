#!/bin/bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda:/usr/local/cuda/lib64:/opt/OpenBLAS/lib

MODEL_PATH="./model"

#INPUT_PATH="../data/aaai19srl.train0.conll.srl.json"
#OUTPUT_PATH="../temp/orl.train0.out"

INPUT_PATH="../data/aaai19srl.dev0.conll.json"
GOLD_PATH="../data/conll_format/aaai19srl.dev0.conll"
OUTPUT_PATH="../temp/orl.devel0.out"

INPUT_PATH="../data/aaai19srl.test0.conll.json"
GOLD_PATH="../data/conll_format/aaai19srl.test0.conll"
OUTPUT_PATH="../temp/orl.test0.out"

ORL_CONS="../data/sentences/orl.2.0.all0.sentences.txt.constituent.txt"
SYS_DEP="../data/dependency_trees/orl.2.0.auto.dep.txt"

CUDA_VISIBLE_DEVICES=$1 python3 ../src/orl-4.1/predict.py \
  --span="span" \
  --model="$MODEL_PATH" \
  --input="$INPUT_PATH" \
  --gold="$GOLD_PATH" \
  --orl_cons=$ORL_CONS \
  --auto_dep_trees=$SYS_DEP  \
  --output="$OUTPUT_PATH" \
  --gpu=$1

