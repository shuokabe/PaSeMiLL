# data directory

## Raw files

`raw_data` contains the raw files, downloaded from their respective websites ([Leipzig news corpora](https://wortschatz.uni-leipzig.de/en/download/German) and [WMT2022 Shared Task in Unsupervised MT and Very Low Resource Supervised MT](https://www.statmt.org/wmt22/unsup_and_very_low_res.html), which also contains data from the previous editions).


More precisely, for Upper Sorbian (HSB):
- monolingual German data: Leipzig news corpus (**2020**, 300K sentences)
- monolingual Upper Sorbian data: dataset from the Sorbian Institute from the **WMT2020** Shared Task (`sorbian_institute_monolingual.hsb.gz`)
- German-Upper Sorbian parallel corpus: combined development and development test datasets from the **WMT2020** Shared Task (`devtest.tar.gz`)

For Lower Sorbian (DSB):
- monolingual German data: Leipzig news corpus (**2022**, 100K sentences)
- monolingual Lower Sorbian data: from the **WMT2022** Shared Task (`66408_DSB_monolingual.txt.gz`)
- German-Lower Sorbian parallel corpus: combined development and development test datasets from the **WMT2022** Shared Task (`valid.de.gz`, `valid.dsb.gz`)

## BUCC-style data files
`bucc_style_data` contains the processed data files, already split into training and test datasets.

The files are named in the following format:
- language pair (e.g., hsb-de)
- training or test dataset (e.g., train)
- file type: language (e.g., hsb or de) or gold sentence pair list (e.g., gold)

as in `hsb-de.train.hsb`.

## Remarks on the pre-training data
Monolingual corpora from the Upper Sorbian section (`monolingual_de-hsb_{de|hsb}.txt`) have been used for the German and Upper Sorbian pre-training (after simple pre-processing).