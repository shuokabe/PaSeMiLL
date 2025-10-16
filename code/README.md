# PaSeMiLL pipeline

The pipeline follows the [UnsupPSE](https://github.com/hangyav/UnsupPSE) pipeline.
1. Convert sentences into vectors.
2. Compute similarity scores between your source and target sentences.
3. Filter the output sentence pairs based on a defined threshold (hyperparameter).

## Detailed pipeline
Inputs (monolingual and one sentence per line):
- Source sentences
- Target sentences

The pipeline relies on the following three steps:


### 1. Converting sentences into vectors

The `contextual_sentence_embeddings.py` file can use the backend language model of your choice (e.g., XLM-R, Glot500, pre-trained, LaBSE, etc.).
Using the following command, you can convert the `input_file.txt` into a file (`output.vec`) with the sentences converted with the `model_name` model.

```
 python contextual_sentence_embeddings.py --input_file input_file.txt --output_file output.vec --model_name model_name
```


### 2. Mining using faiss

You can mine from the two created embedding files using the following terminal command:
```
python scripts/bilingual_nearest_neighbor.py --source_embeddings source_embeddings.vec --target_embeddings target_embeddings.vec --output output_dictionary.sim --knn 10 -m csls --cslsknn 20 
```
This will take `source_embeddings.vec` and `target_embeddings.vec` as input files and output a dictionary `output_dictionary.sim`.
It uses the CSLS metrics to compute the similarity (`-m csls`).

We run this part with the following packages:
- faiss-gpu: 1.7.1
- numpy: 1.19.2
- nltk: 3.6.5
- gensim: 3.2.0
- tqdm: 4.63.0
- torch: 1.13.1
- transformers: 4.30.2

### 3. Filtering the mined pairs

The final filtering is done using the `filter.py` file. 
It takes the previous similarity dictionary `output_dictionary.sim` as input and outputs sentence pairs (stored in `output.sim.pred`) with a score above a `dynamic` threshold.
```
python scripts/filter.py --input output_dictionary.sim --output output.sim.pred -m dynamic -th 2.0
```
When a file with the gold sentence pairs (`sentence_pair.gold`) exists, you can evaluate the mining quality with the usual precision, recall, and F-score, as in the corresponding BUCC Shared Task.
```
python scripts/bucc_f-score.py -p output.sim.pred -g sentence_pair.gold > output.sim.pred.res
```

## Optional commands
#### Alignment post-processing
```
python align_source_target.py -m output.sim.pred -s source_file -t target_file -o aligned.sim -l model_name
```
After a first filtering based on the similarity score (`output.sim.pred`), alignment links are computed between the original source and target sentences (`source_file` and `target_file`) with SimAlign.

#### CBIE
```
outlier_normalisation.py -m model_name -s src_lang -t trg_lang
```
This code requires changing the file paths in the code to run.