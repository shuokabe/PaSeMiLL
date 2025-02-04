#!/bin/bash

cd /home_test # TO CHANGE: folder path

python ./pretraining/tokenisation/run.py \
  --input_fname ./data/conc_90_hsb_cs_de.txt \
  --model_name xlm-roberta-base \
  --save_directory ./pretraining/tokenisation/ \
  --vocab_size 120000 

# Parameter to change:
# input_fname: Data on which to train the tokeniser