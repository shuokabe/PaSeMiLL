# Pre-training XLM-R with Upper Sorbian data

The core code for this part comes from the [Glott500 repository](https://github.com/cisnlp/Glot500/), which itself follows the methodology from HuggingFace's Transformers library (see [here](https://github.com/huggingface/transformers/)).

There are two steps to pre-train a language model using data from an unseen language:
1. Training a tokeniser for the language to add (cf. `tokenisation` folder)
2. Actual pre-training of the language model with the data (cf. `modelling` folder)

Parameters in shell scripts must be modified to match the data path.