# Causal-BERT
Causality-detection model based on BERT-embeddings trained on historical German texts.
Code to accompany my master's thesis ![Causal Relationship Extraction from Historical Texts using BERT](/docs/thesis.pdf)

![Example](/docs/example.png)

## Installation
Install dependencies in virtual env.

    python3 -m virtualenv env
    source env/bin/activate
    pip install requirements.txt

Install spacy models

    python -m spacy download de_core_news_md
    python -m spacy download en_core_web_sm

## Experiments

Change to source folder

    cd /src

To run a test run of the experiments.

    python main.py --pretrain True --debug True 

To run all experiments without input modification.

    python main.py --pretrain True

To run all experiments with text normalization.

    python main.py --normalize True --pretrain False

To run all experiments with coreference information.

    python main.py --coref True --pretrain False

"--pretrain False" can be used if there already exist pre-trained models.

## Interactive Usage

The notebook `scripts/practical_usage.ipynb` shows an usage example on raw pdf files which can be used for further tasks and to help adapt the code to other applications.

## Model Architecture

The model uses BERT embeddings in a series of sequential tasks to detect causal relations.

![Model Architecture](/docs/model_structure.png)
