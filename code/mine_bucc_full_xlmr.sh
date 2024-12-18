#!/bin/bash
cd home_test/UnsupPSE # TO CHANGE: folder path

pip install sentencepiece
pip install sentence-transformers # For LaBSE
#pip install protobuf==3.20.0 # For glot500
#pip install gensim # When using latest conda environment. WARNING: scipy has been downgraded when installing gensim!
#pip install tokenizers -U

#source ./environment.sh
source ./environment_full_xlmr.sh

# TO CHANGE: model name prefix
# Do not forget the . after the model name for all models except the base one
PREFIX='distil_xlmr.'  #'laser3.' #'glot500.' #'laser.' #'pretrained-hsb.' #'glot500.' #'LaBSE.'

# ================= MKDIR =====================
#for src_lang in $SRC_LANGS; do
#	for trg_lang in $TRG_LANGS; do
#		mkdir -p $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang
#		mkdir -p $MINING/bucc2017/$src_lang-$trg_lang
#	done;
#done;
#mkdir -p $DICTIONARIES

# ================= AVERAGED DOCUMENT REPRESENTATION ===============
# With XLM-R representations directly (not tokenised and true cased)
#for data in $BUCC_SETS; do
#    for src_lang in $SRC_LANGS; do
#        for trg_lang in $TRG_LANGS; do
#            for lang in $src_lang $trg_lang; do
#		echo "$src_lang-$trg_lang document embeddings";
#                $PYTHON contextual_document_embeddings.py --input_file $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$lang \
#                --output_file $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$lang.vec -m 'laser'
#            done;
#        done;
#    done;
#done;

# ================= DOC_DICTIONARY_GENERATION_FOR_PREFILTERING ===============
for data in $BUCC_SETS; do
    for src_lang in $SRC_LANGS; do
        for trg_lang in $TRG_LANGS; do
		echo "$src_lang-$trg_lang nearest neighbour";
		CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/bilingual_nearest_neighbor.py --source_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$src_lang.vec \
		--target_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$PREFIX$src_lang-$trg_lang.$data.$trg_lang.vec \
		--output $DICTIONARIES/DOC.$PREFIX$src_lang-$trg_lang.$data.sim --knn 10 -m csls --cslsknn 20 #nn #$PREFIX
                #--output $DICTIONARIES/DOC.$src_lang-$trg_lang.$data.sim --knn $TOPN_DOC -m csls --cslsknn $TOPN_CSLS #nn
            
		# Opposite direction
		#CUDA_VISIBLE_DEVICES=$GPUS $PYTHON scripts/bilingual_nearest_neighbor.py --source_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$trg_lang.vec \
                #--target_embeddings $DOC_EMBEDDINGS/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.$src_lang.vec \
                #--output $DICTIONARIES/DOC.$trg_lang-$src_lang.$data.sim --knn 10 -m csls --cslsknn 20
        done;
    done;
done;

# ================= MINING_&_EVALUATION ==============================
#for mine_method in $MINE_METHODS; do
#    for filter_method in $FILTER_METHODS; do
#        for data in $BUCC_SETS; do
#            for src_lang in $SRC_LANGS; do
#                for trg_lang in $TRG_LANGS; do
#                    $PYTHON ./scripts/filter.py -i $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_$src_lang-$trg_lang.$data.sim -m $filter_method -th ${FILTER_THRESHOLDS[bucc17_${mine_method}_${filter_method}_${src_lang}_${trg_lang}]} \
#			-o $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred
#                    $PYTHON scripts/bucc_f-score.py  -p $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred -g $DATA/bucc2017/$src_lang-$trg_lang/$src_lang-$trg_lang.$data.gold > $MINING/bucc2017/$src_lang-$trg_lang/${mine_method}_${filter_method}_$src_lang-$trg_lang.$data.sim.pred.res
#                done;
#            done;
#        done;
#    done;
#done;
