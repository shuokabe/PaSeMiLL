#!/bin/bash

#pip install transformers
#pip install sentencepiece
#pip install accelerate -U
pip install tokenizers==0.19.0

cd /home_test/ # TO CHANGE: folder path

torchrun ./pretraining/modelling/run_mlm.py \
    --model_name_or_path xlm-roberta-base \
    --train_file ./data/conc_90_hsb_cs_de.txt \
    --tokenizer_name ./pretraining/tokenisation/Glot500_extended_spm \
    --per_device_train_batch_size 8 \
    --do_train \
    --output_dir ./pretraining/modelling/output-hsb-para-cs/ \
    --save_steps 10000 \
    --num_train_epochs 5 \
    --line_by_line true

# Parameters to change:
# train_file: File used for pre-training (e.g., German and Upper Sorbian corpus)
# tokenizer_name: tokeniser trainined with the tokenisation code
# output_dir: folder path for the output model (and related output files, checkpoints)