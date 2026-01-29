# PaSeMiLL: Parallel Sentence Mining for Low-Resource Languages

This repository contains the published artefacts for parallel sentence mining from ([Okabe and Fraser, 2025](https://aclanthology.org/2025.computel-main.2.pdf)) and ([Okabe et al., 2025](https://aclanthology.org/2025.acl-short.17.pdf)).

## Data section
The `data` folder contains the raw (unprocessed) datasets and the BUCC-style files for Upper and Lower Sorbian for (Okabe and Fraser, 2025).

The dataset from (Okabe et al., 2025) is available on its dedicated repository: [Belopsem](https://github.com/shuokabe/Belopsem).

## Code section
The `code` folder contains useful code to use with the original [UnsupPSE](https://github.com/hangyav/UnsupPSE) pipeline ([Hangya and Fraser, 2019](https://aclanthology.org/P19-1118.pdf)).

The pre-training of XLM-R with Upper Sorbian data is addressed in the `pretraining` subfolder.

## How to use?
The full pipeline is available in the `mine_bucc_full_xlmr.sh` file.

Once your two monolingual corpora are ready (one sentence per line):
1. Convert the sentences into embeddings using the backend language model of your choice (e.g., XLM-R, Glot500, or pre-trained).
2. Compute similarity scores between your source and target sentences.
3. Filter the output sentence pairs based on a defined threshold (hyperparameter).

## Citations
For the updated version of the pipeline, please use the following citation (from the ACL Anthology):

```
@inproceedings{okabe-etal-2025-improving,
    title = "Improving Parallel Sentence Mining for Low-Resource and Endangered Languages",
    author = {Okabe, Shu  and
      H{\"a}mmerl, Katharina  and
      Fraser, Alexander},
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-short.17/",
    doi = "10.18653/v1/2025.acl-short.17",
    pages = "196--205",
    ISBN = "979-8-89176-252-7",
}
```

If you use the BUCC-style dataset from the `dataset` folder or the older version of the pipeline, please use the following citation (from the ACL Anthology):

```
@inproceedings{okabe-fraser-2025-bilingual,
    title = "Bilingual Sentence Mining for Low-Resource Languages: a Case Study on Upper and {L}ower {S}orbian",
    author = "Okabe, Shu  and
      Fraser, Alexander",
    editor = "Lachler, Jordan  and
      Agyapong, Godfred  and
      Arppe, Antti  and
      Moeller, Sarah  and
      Chaudhary, Aditi  and
      Rijhwani, Shruti  and
      Rosenblum, Daisy",
    booktitle = "Proceedings of the Eight Workshop on the Use of Computational Methods in the Study of Endangered Languages",
    month = mar,
    year = "2025",
    address = "Honolulu, Hawaii, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.computel-main.2/",
    pages = "11--19",
}
```

## Acknowledgements

This work has received funding from the European Research Council (ERC) under grant agreement No. 101113091 - Data4ML, an ERC Proof of Concept Grant.

