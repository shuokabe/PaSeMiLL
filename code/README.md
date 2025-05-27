# PaSeMiLL pipeline


2. Compute similarity scores between your source and target sentences.
3. Filter the output sentence pairs based on a defined threshold (hyperparameter).

Inputs (monolingual and one sentence per line):
- Source sentences
- Target sentences

The pipeline relies on the three following steps:


1. Converting sentences into vectors


The `contextual_sentence_embeddings.py` file can use the backend language model of your choice (e.g., XLM-R, Glot500, pre-trained, LaBSE, etc.).

2. Mining using faiss

You can mine from the two created embedding files using the following terminal command:
```
python scripts/bilingual_nearest_neighbor.py --source_embeddings source_embeddings.vec --target_embeddings target_embeddings.vec --output output_dictionary.sim --knn 10 -m csls --cslsknn 20 
```
This will take `source_embeddings.vec` and `target_embeddings.vec` as input files and output a dictionary `output_dictionary.sim`.
It uses the CSLS metrics to compute the similarity (`-m csls`).

3. Filtering the mined pairs

The final filtering is done using the `filter.py` file.