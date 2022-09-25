import datetime
import os

import pandas as pd
import spacy
import torch
from datasets import load_metric
from tqdm import tqdm

from helper_functions import (get_dunietz_data, get_fondsforste_data, get_forstvermessung_data, 
                              get_Rehbein_data, dunietz_sanity_check, rehbein_sanity_check, 
                              fondsforste_sanity_check, forstvermessung_sanity_check)
from training_utils import pre_train_models, run_cross_validation_for_models, get_kfolds

pd.set_option('display.max_columns', None)
tqdm.pandas()

PROJECT_PATH = "/content/drive/My Drive/master thesis/"
FONDSFORSTE_DATSET_PATH = "/content/drive/My Drive/master thesis/data/fondsforste/"
FORSTVERMESSUNG_DATSET_PATH = "/content/drive/My Drive/master thesis/data/forstvermessung/"
REHBEIN_DATA_PATH = "/content/drive/My Drive/master thesis/data/causal_language_DE_release1.0/"
DUNIETZ_DATA_PATH = "/content/drive/My Drive/master thesis/data/BECAUSE/"
CURRENT_TIME = datetime.datetime.now().strftime('%Y_%m_%d_%H%M')

causal_arguments = ["Cause", "Effect", "Affected", "Actor", "Controlling", "Support", "Trigger"]
entity_options = {
    'colors': {
        'Cause': '#99FCE0',
        'Effect': '#6779CB',
        'Affected': '#84F72D',
        'Actor': '#108482',
        'Controlling': '#E3AF32',
        'Support': '#C44C6D',
        'Trigger': '#C5E95E',
    },
    'ents': causal_arguments
}

config = {
    "seed": 42,
    "test_size": 0.2,
    "val_size": 0.1,
    "results_directory": PROJECT_PATH + f"output/{CURRENT_TIME}/",
    "rehbein_model_file_path": PROJECT_PATH + "models/pretrained/CBERT_rehbein",
    "dunietz_model_file_path": PROJECT_PATH + "models/pretrained/CBERT_dunietz",
    "rehbein_dunietz_model_file_path": PROJECT_PATH + "models/pretrained/CBERT_rehbein_dunietz",
    "fondsforste_forstvermessung_rehbein_dunietz_model_file_path": PROJECT_PATH + "models/pretrained/CBERT_fondsforste_forstvermessung_rehbein_dunietz",
    "retrain_pretrained": False,
    "model_checkpoints": [
                        #   "dbmdz/bert-base-german-uncased", 
                          "dbmdz/bert-base-german-cased",
                          "dbmdz/bert-base-german-europeana-cased",
                          "bert-base-multilingual-cased",
    ],
    "batch_size": 2,
    "max_length": 500,
    "num_epochs": 75,
    "learning_rate": 2.5e-5,
    "bert_embedding_size": 768,
    "bert_dropout": 0.1,
    "early_stop_patience": 5,
    "causal_arguments": causal_arguments,
    "nlp_ger": spacy.load('de_core_news_md'),
    "nlp_eng": spacy.load("en_core_web_sm"),
    "entity_options": entity_options,
    "val_epochs": 5,
    "add_coreference": False,
    "debug": True,
    "time": CURRENT_TIME,
    "use_normalized": False,
    "warmup_steps": 1_000,
    "label_dict": {
        'B-Trigger': 0,
        'O': 1, # "Trigger" and "O" 012 so that Trigger-Detection task is automatically aligned
        'B-Actor': 2,
        'I-Actor': 3,
        'B-Affected': 4,
        'I-Affected': 5,
        'B-Cause': 6,
        'I-Cause': 7,
        'B-Controlling': 8,
        'I-Controlling': 9,
        'B-Effect': 10,
        'I-Effect': 11,
        'B-Support': 12,
        'I-Support': 13,
    },
    "type_dict": {
        "Purpose": 0,
        "Motivation": 1,
        "Consequence": 2,
        "None": 3,
    },
    "degree_dict": {
        "Facilitate": 0,
        "Inhibit": 1,
        "None": 2,
    },
    "strict_metric": load_metric("seqeval"),
    "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
}
    
if config["debug"]:
    config["num_epochs"] = 1
    config["model_checkpoints"] = config["model_checkpoints"][:1]
    config["results_directory"] = config["results_directory"][:-1] + "_debug/"
    config["rehbein_model_file_path"] = config["rehbein_model_file_path"] + "_debug"
    config["dunietz_model_file_path"] = config["dunietz_model_file_path"] + "_debug"
    config["rehbein_dunietz_model_file_path"] = config["rehbein_dunietz_model_file_path"] + "_debug"
    config["fondsforste_forstvermessung_rehbein_dunietz_model_file_path"] = config["fondsforste_forstvermessung_rehbein_dunietz_model_file_path"] + "_debug"

try:
    os.mkdir(config["results_directory"])
except:
    print(f"Model directory {config['results_directory']} already created")

config["label_list"] = list(config["label_dict"].keys())
config["type_list"] = list(config["type_dict"].keys())
config["degree_list"] = list(config["degree_dict"].keys())

if __name__ == "__main__":

    # get data
    fondsforste_sentence_data = get_fondsforste_data(FONDSFORSTE_DATSET_PATH, config)
    forstvermessung_sentence_data = get_forstvermessung_data(FORSTVERMESSUNG_DATSET_PATH, config)
    rehbein_sentence_data = get_Rehbein_data(REHBEIN_DATA_PATH, config)
    dunietz_sentence_data = get_dunietz_data(DUNIETZ_DATA_PATH, config)#

    # sanity checks
    fondsforste_sanity_check(fondsforste_sentence_data)
    forstvermessung_sanity_check(forstvermessung_sentence_data)
    rehbein_sanity_check(rehbein_sentence_data)
    dunietz_sanity_check(dunietz_sentence_data)

    # combine data
    evaluation_sentence_data = fondsforste_sentence_data + forstvermessung_sentence_data


    # optional pre-training
    if config["retrain_pretrained"]:
        dunietz_model_results = pre_train_models(
            dunietz_sentence_data, 
            config["dunietz_model_file_path"], 
            config
        )
        display(dunietz_model_results)

        rehbein_model_results = pre_train_models(
            rehbein_sentence_data, 
            config["rehbein_model_file_path"], 
            config
        )
        display(rehbein_model_results)

        rehbein_dunietz_model_results = pre_train_models(
            rehbein_sentence_data+dunietz_sentence_data, 
            config["rehbein_dunietz_model_file_path"],
            config,
        )
        display(rehbein_dunietz_model_results)

        fondsforste_forstvermessung_rehbein_dunietz_model_results = pre_train_models(
            rehbein_sentence_data+dunietz_sentence_data+evaluation_sentence_data, 
            config["fondsforste_forstvermessung_rehbein_dunietz_model_file_path"],
            config,
        )
        display(fondsforste_forstvermessung_rehbein_dunietz_model_results)


    # Run Experiments
    eval_kfold = get_kfolds(evaluation_sentence_data, config)

    model_results_no_transfer = run_cross_validation_for_models(
        evaluation_sentence_data, 
        kfold_df=eval_kfold,
        pretrained_model_path_base=None, 
        model_name="no_transfer", 
        corpus_name="evaluation_data",
        debug=True,
        config=config,
    )

    model_results_rehbein = run_cross_validation_for_models(
        evaluation_sentence_data, 
        kfold_df=eval_kfold,
        pretrained_model_path_base=config["rehbein_model_file_path"], 
        model_name="rehbein", 
        corpus_name="evaluation_data",
        config=config,
    )

    model_results_dunietz = run_cross_validation_for_models(
        evaluation_sentence_data, 
        kfold_df=eval_kfold,
        pretrained_model_path_base=config["dunietz_model_file_path"], 
        model_name="dunietz", 
        corpus_name="evaluation_data",
        config=config,
    )

    model_results_rehbein_dunietz = run_cross_validation_for_models(
        evaluation_sentence_data, 
        kfold_df=eval_kfold,
        pretrained_model_path_base=config["rehbein_dunietz_model_file_path"], 
        model_name="rehbein_dunietz", 
        corpus_name="evaluation_data",
        config=config,
    )


    print("Success")

