# PaSeMiLL: Parallel Sentence Mining for Low-Resource Languages

## Data section
The `data` folder contains the raw (unprocessed) datasets and the BUCC-style files for Upper and Lower Sorbian.

## Code section
The `code` folder contains useful code to use with the original [UnsupPSE](https://github.com/hangyav/UnsupPSE) pipeline ([Hangya and Fraser, 2019](https://aclanthology.org/P19-1118.pdf)).

The pre-training of XLM-R with Upper Sorbian data is addressed in the `pretraining` subfolder.

## How to use?
The full pipeline is available in the `mine_bucc_full_xlmr.sh` file.

Once your two monolingual corpora are ready (one sentence per line):
1. Convert the sentences into embeddings using the backend language model of your choice (e.g., XLM-R, Glot500, or pre-trained).
2. Compute similarity scores between your source and target sentences.
3. Filter the output sentence pairs based on a defined threshold (hyperparameter).