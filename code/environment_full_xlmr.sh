# File modified from https://github.com/hangyav/UnsupPSE/blob/master/environment.sh
# ================= BASH =======================================================
# print commands
set -x
# exit in case of error
set -e
# ================= EXECUTABLES ================================================
PYTHON=python
# PYTHON=~/.anaconda/envs/pse/bin/python
FASTTEXT=./third_party/fastText/fasttext
MUSE=./third_party/MUSE
MOSES=./third_party/moses/scripts

# ================= DATA =======================================================
DATA=./data
# IN CASE OF OTHER LANGUAGES than en de fr ru make sure to download monolingual data and preprocess them like in get_data.sh
# INPUT=new_news.2011-14 # data for creating dictionaries; this is extended with src and trg language ids

# PARAMETERS FOR BUCC EXPERIMENTS
SRC_LANGS='oci' # #'chv'  #'hsb' # dsb' # 'cs' #'de' # 'dsb' #'hsb' #de fr ru' #de # which BUCC language pairs to run; each source language is paired with each target language
TRG_LANGS='es' #'ka' #'fr' #'en' #'es' #'ru' #'de' #'en' #en
BUCC_SETS='train test' 
PREFIX='xlmr.' #'glot500.' #'LaBSE.'
# For BUCC datasets, use 'sample training', otherwise, 'train test'

RESULTS=./results_full_xlmr
EMBEDDINGS=$RESULTS/new_embeddings #embeddings
# BWE_EMBEDDINGS=$EMBEDDINGS/bwe
DOC_EMBEDDINGS=$EMBEDDINGS/doc
DICTIONARIES=$RESULTS/dictionaries
MINING=$RESULTS/mining
FILTERING=$RESULTS/filtering

# ============= MINING PARAMETERS =============================================
THREADS=20 # number of threads to use
GPUS=0 # comma separated list of visible GPU indices
DIM=768 #300 # dimension of embeddings
TOPN_DICT=100 # topn most similar translation candidates in the generated BWE dictionary
TOPN_CSLS=500 # topn most similar neighbors used for CSLS
TOPN_DOC=100 # topn most similar document candidates for prefiltering

# BUCC
# MINE_METHODS='maxalign max' # list of mining methods; supported: max (averaging most similar words); maxalign (subsegment detection)
# FILTER_METHODS='static dynamic' # list of filtering methods; supported: static; dynamic
