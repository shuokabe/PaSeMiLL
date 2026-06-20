# File modified from https://github.com/hangyav/UnsupPSE/blob/master/environment.sh
# ================= BASH =======================================================
# print commands
set -x
# exit in case of error
set -e
# ================= EXECUTABLES ================================================
PYTHON=python
# PYTHON=~/.anaconda/envs/pse/bin/python

# ================= DATA =======================================================
DATA=./data

# PARAMETERS FOR BUCC EXPERIMENTS
SRC_LANGS='oci' #'chv' #'hsb' # 'dsb' 'oci' # which BUCC language pairs to run; each source language is paired with each target language
TRG_LANGS='es' #'en' #'es' #'ru' #'de' 
BUCC_SETS='train test' 
MODELS='xlmr' # glot500 LaBSE' #'glot500.' #'LaBSE.'
# For BUCC datasets, use 'sample training', otherwise, 'train test'

# TO UPDATE
RESULTS=./results_full_xlmr
EMBEDDINGS=$RESULTS/new_embeddings 
DOC_EMBEDDINGS=$EMBEDDINGS/doc
DICTIONARIES=$RESULTS/dictionaries
MINING=$RESULTS/mining
