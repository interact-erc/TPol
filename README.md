# TPol - Translate First Reorder Later
This is the repo for paper [EACL23 Findings paper](https://arxiv.org/abs/2210.04878).

## Installation
First, create a Conda environment using the tpol-environment.yml file and activate it using the following commands:
```
conda env create -f tpol-environment.yml
conda activate tpol-private
```

## Instructions

1. Import the data: clone the [GEO-Aligned](https://github.com/interact-erc/GEO-Aligned) repository in `data/`:
```
$ cd data/
$ git clone https://github.com/interact-erc/GEO-Aligned.git
$ cd ..
```
2.Run the Lexical Translator script. You have the option to choose between the bert and the mbart approach. For instance, to run the bert approach, execute the zbert_translator.pyz script.
3. Run the MR Reorderer script. Similarly, you can choose between the bert and the mbart approach. For instance, to run the mbart approach, execute the `mbart_reorderer.py` script.

Both the translator and reorderer scripts require the following arguments:
```
- --dataset: path to the dataset file
- --test-ids: path to the test ids file
- --dev-ids: path to the dev ids file
- --language: dataset language, choose among en, it, de
- --out-file: path to the file where the test predictions are be saved
- --results-file: path to the file where the numerical results are saved
```

Additionally, the reorderer script requires the --lexical-predictions argument, which should be the file outputted by the translator --out-file.

To run mbart_reorderer in silver mode, first run the translator with the additional --all-predictions-file argument (the path to the file storing the predictions). Then, provide the same file to the reorderer as the --silver-predictions-path.
