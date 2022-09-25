import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from helper_functions import (NumpyEncoder,
                              filter_metrics_from_validation_results, flatten,
                              get_kfolds, get_labels_for_sentence,
                              get_test_predictions, postprocessing_predictions,
                              print_dataset_statistics, seed_everything,
                              get_argument_sample_result, get_dummy_result)
from metrics import (calc_relation_metrics, get_argument_detection_metrics,
                     get_combine_trigger_metrics, get_degree_metrics,
                     get_trigger_metrics, get_type_metrics)
from model import CBERT
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_scheduler


class CBERTDataset(Dataset):
    def __init__(self, sentence_ids, sentence_data, tokenizer, config, only_causal=False, add_coreferences=False, use_normalized=False):
        """Dataset for CBERT model.

        Parameters
        ----------
        sentence_ids : list
            Ids of sentence data used in this dataset
        sentence_data : list
            List of sentence samples with tokens and relations
        tokenizer : Huggingface Tokenizer
            Tokenizer for the used BERT model
        config : dict
            Dictionary containing overall parameters and constants
        only_causal : bool, optional
            Whether only causal samples are returned, by default False
        add_coreferences : bool, optional
            Whether coreference labels are added, by default False
        use_normalized : bool, optional
            Whether normalized tokens are returned, by default False
        """
        self.sentence_ids = sorted(sentence_ids)
        self.sentence_data = sentence_data
        self.tokenizer = tokenizer
        self.config = config

        self.add_coreferences = add_coreferences
        self.use_normalized = use_normalized

        if only_causal:
            self.sentence_ids = [id for id in self.sentence_ids if self.sentence_data[id]["is_causal"]]


    def __len__(self):
        return len(self.sentence_ids)


    def __gettokens__(self, sent_id):

        if self.use_normalized:
            return self.sentence_data[sent_id]["normalized_tokens"]
        else:
            return self.sentence_data[sent_id]["tokens"]


    def __getitem__(self, idx):

        # only use ids of available sentences
        sent_id = self.sentence_ids[idx]

        sent_data = self.sentence_data[sent_id]
        relations = sent_data["relations"]
        tokens = self.__gettokens__(sent_id)
        labels = get_labels_for_sentence(tokens, relations, self.add_coreferences)
        tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=self.config["max_length"])
        
        # make subword_ids
        word_ids = np.array(tokenized_inputs.word_ids(batch_index=0))
        word_ids[[0, -1]] = -1, len(tokens)+1

        sample_subwords = []
        for i in pd.Series(word_ids).value_counts().sort_index().values:
            sample_subwords += list(range(i))

        # CLS and SEP do not count
        sample_subwords[0] = -1
        sample_subwords[-1] = -1

        # construct tag set for input_ids
        tags = []
        for labels_rel in labels:
            tags_rel = [self.config["label_dict"][i] for i in labels_rel]

            word_ids = tokenized_inputs.word_ids(batch_index=0)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(tags_rel[word_idx])
                # For the other tokens in a word, we set the label to the current label
                else:
                    label_ids.append(tags_rel[word_idx])
                previous_word_idx = word_idx

            tags.append(label_ids)

        # pad tags to equal size
        sample_relation_arguments = np.array(tags + [[-100] * len(tokenized_inputs["input_ids"]) for _ in range(16-len(tags))]) # 16 max number of relations per sentence, -100 as code for unused relation

        # construct tags with Triggers of all relations in sample (for Task 1)
        sample_triggers = sample_relation_arguments[0].copy()
        sample_triggers[(sample_triggers != self.config["label_dict"]["B-Trigger"])] = self.config["label_dict"]["O"]
        sample_triggers[np.sum(sample_relation_arguments == self.config["label_dict"]["B-Trigger"], axis=0) > 0] = self.config["label_dict"]["B-Trigger"]
        sample_triggers[[0, -1]] = -100

        # construct causal type and degree labels for relations
        sample_relation_types = [self.config["type_dict"][rel["type"]] for rel in relations]
        sample_relation_types = np.array(sample_relation_types + [-100] * (16-len(sample_relation_types))) # pad to same number as relations

        sample_relation_degrees = [self.config["degree_dict"][rel["degree"]] for rel in relations]
        sample_relation_degrees = np.array(sample_relation_degrees + [-100] * (16-len(sample_relation_degrees))) # pad to same number as relations

        return {
            "input_ids": tokenized_inputs["input_ids"], 
            "attention_masks": tokenized_inputs["attention_mask"], 
            "sample_subwords": sample_subwords,
            "sample_relation_arguments": sample_relation_arguments, 
            "sample_triggers": sample_triggers,
            "sample_relation_types": sample_relation_types, 
            "sample_relation_degrees": sample_relation_degrees, 
            "sent_id": sent_id
        }


    def collate_fn(self, batch):
        """Padds samples in batch to equal size and converts them to tensor.


        Parameters
        ----------
        batch : 
            Samples in a batch.

        Returns
        -------
        Dict of torch.tensor
            Contains padded tensors of sample data 
        """
        max_length = max([len(sample["input_ids"]) for sample in batch])

        new_batch = []
        for sample in batch:
            
            len_to_pad = max_length - len(sample["input_ids"])
            input_ids = sample["input_ids"] + [self.tokenizer.pad_token_id]*len_to_pad
            attention_masks = sample["attention_masks"] + [0]*len_to_pad
            sample_subwords = sample["sample_subwords"] + [-1]*len_to_pad
            sample_relation_arguments = [trigger_inds.tolist() + [-100]*len_to_pad for trigger_inds in sample["sample_relation_arguments"]]
            sample_triggers = sample["sample_triggers"].tolist() + [-100]*len_to_pad

            new_sample = {
                "input_ids": torch.tensor(input_ids),
                "attention_masks": torch.tensor(attention_masks),
                "sample_subwords": torch.tensor(sample_subwords),
                "sample_relation_arguments": torch.tensor(sample_relation_arguments),
                "sample_triggers": torch.tensor(sample_triggers),
                "sample_relation_types": torch.tensor(sample["sample_relation_types"]),
                "sample_relation_degrees": torch.tensor(sample["sample_relation_degrees"]),
                "sent_id": torch.tensor(sample["sent_id"]),
            }

            new_batch.append(new_sample)
        
        collated_batch = {}
        for key in batch[0].keys():
            collated_samples = [sample[key] for sample in new_batch]
            collated_batch[key] = torch.stack(collated_samples)

        return collated_batch

# Original Source https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, score, model):

        if score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation metric increased ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)


def train_model(
    train_dataloader, 
    val_dataloader, 
    model_checkpoint, 
    config,
    num_epochs=None,
    learning_rate=None,
    model_path=None,
    early_stopping=False, 
    pretrained_model_path=None, 
    pretrained_bert_model_path=None, 
    debug=False, 
):
    """Main Training Function

    Parameters
    ----------
    train_dataloader : torch.DataLoader
        Dataloader with Training Samples
    val_dataloader : torch.DataLoader
        Dataloader with Validation Samples
    model_checkpoint : string
        Huggingface Model Name
    config : dict
        Dictionary containing overall parameters and constants
    num_epochs : int, optional
        Number of epochs to train
        If not given, use value in config, by default None
    learning_rate : float, optional
        Learning rate during training
        If not give, use value in config, by default None
    model_path : string, optional
        Location where the model is saved, by default None
    early_stopping : bool, optional
        Whether early stoppping is performed, by default False
    pretrained_model_path : string, optional
        Location of pre-trained model weights
        If None, no weights are loaded, by default None
    pretrained_bert_model_path : string, optional
        Location of pre-trained BERT weights
        If None, no weights are loaded, by default None
    debug : bool, optional
        Whether debug output is provided, by default False

    Returns
    -------
    pd.DataFrame, torch.nn.Module
        Aggregated validation results and trained model
    """
    if num_epochs == None: num_epochs = config["num_epochs"]
    if learning_rate == None: learning_rate = config["learning_rate"]

    model = CBERT(model_checkpoint, config)

    if pretrained_bert_model_path:
        print(f"Used pretrained bert model from {pretrained_bert_model_path}")
        model.load_state_dict(torch.load(pretrained_bert_model_path), strict=False)

    if pretrained_model_path:
        print(f"Used pretrained model from {pretrained_model_path}")
        model.load_state_dict(torch.load(pretrained_model_path))

    model.to(device=config["device"])
    num_training_steps = num_epochs * len(train_dataloader)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    scaler = torch.cuda.amp.GradScaler()

    if early_stopping:
        es = EarlyStopping(patience=config["early_stop_patience"], path=model_path, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        losses = {
            "triggers_loss": 0,
            "combine_triggers_loss": 0,
            "args_loss": 0,
            "type_loss": 0,
            "degree_loss": 0,
        }
        for batch in train_dataloader:
            _, batch_losses = model(
                    input_ids=batch["input_ids"].to(config["device"]), 
                    attention_masks=batch["attention_masks"].to(config["device"]), 
                    subwords=batch["sample_subwords"].to(config["device"]), 
                    triggers=batch["sample_triggers"].to(config["device"]),
                    arguments=batch["sample_relation_arguments"].to(config["device"]),
                    types=batch["sample_relation_types"].to(config["device"]),
                    degrees=batch["sample_relation_degrees"].to(config["device"]),
            )  

            combined_loss = sum(batch_losses.values())
            scaler.scale(combined_loss).backward()
            scaler.unscale_(optimizer) # needed for clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)

            # only update lr if scaling has been done
            scale = scaler.get_scale()
            scaler.update()
            skip_lr_sched = (scale != scaler.get_scale())
            
            if not skip_lr_sched:
                lr_scheduler.step()

            # Save losses
            for loss_name, loss in batch_losses.items():
                losses[loss_name] += loss.item() / len(train_dataloader)

            optimizer.zero_grad()
            progress_bar.update(1)

        if epoch % config["val_epochs"] == 0 or epoch == (config["num_epochs"]-1):
            results = validate(model, val_dataloader, debug=debug, config=config)

            if debug:
                filtered_results = filter_metrics_from_validation_results(results, config)
                filtered_results.index = [epoch]
                display(filtered_results)
                display(pd.DataFrame(losses, index=[f"{epoch} Train Losses"]))

            if early_stopping:
                trigger_arg_meanf1 = (results["detect_args_results"]["overall_f1_strict"] + results["detect_trigger_results"]["Trigger_strict_f1"]) / 2
                es(trigger_arg_meanf1, model)

                if es.early_stop:
                    print(f"Stop Early at epoch {epoch-es.counter}")
                    break
    
    return filter_metrics_from_validation_results(results, config), model


def pre_train_models(sentence_data, model_path_base, config, pretrained_model_path=None):
    """Trains pre-trained models for each model checkpoint in config.
    Also allow to continue training an already pre-trained model (useful if 
    iteratively pre-training on different datasets).

    Parameters
    ----------
    sentence_data : list
        List of sentence samples with tokens and relations
    model_path_base : str
        First part of the location where the trained model will be saved
        (Second part is the model checkpoint) 
    config : dict
        Dictionary containing overall parameters and constants
    pretrained_model_path : string, optional
        Location of pre-trained model weights
        If None, no weights are loaded, by default None

    Returns
    -------
    pd.DataFrame
        Aggregated results of pre-trained models
    """

    model_results = []
    for model_checkpoint in config["model_checkpoints"]:
        
        model_save_path = f"{model_path_base}_{model_checkpoint.replace('/', '_')}.pth"
        results, _ = pre_train_model(sentence_data, model_checkpoint, model_save_path, config, pretrained_model_path=pretrained_model_path)
        model_results.append(results)

    results_combined = pd.concat(model_results)
    results_combined.index = config["model_checkpoints"]

    return results_combined


def pre_train_model(sentence_data, model_checkpoint, model_save_path, config, pretrained_model_path=None):
    """Pretrains a single model given by model_checkpoint on sentence_data with
    early stopping.

    Parameters
    ----------
    sentence_data : list
        List of sentence samples with tokens and relations
    model_checkpoint : string
        Huggingface Model name
    model_save_path : string
        Location where the model is saved
    config : dict
        Dictionary containing overall parameters and constants
    pretrained_model_path : string, optional
        Location of pre-trained model weights
        If None, no weights are loaded, by default None

    Returns
    -------
    pd.DataFrame, torch.nn.Module
        Aggregated validation results and trained model
    """
    
    print(f"Save model to {model_save_path}")
    seed_everything()

    kfold_data = get_kfolds(sentence_data, config, n_splits=10, debug=False)
    kfold_data["sentence_id"] = np.arange(len(kfold_data))      

    train_ids = kfold_data.query("fold != 0")["sentence_id"].to_list()
    test_ids = kfold_data.query("fold == 0")["sentence_id"].to_list()

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # make datasets and loaders
    train_dataset = CBERTDataset(train_ids, sentence_data, tokenizer, config, only_causal=True)
    test_dataset = CBERTDataset(test_ids, sentence_data, tokenizer, config)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=test_dataset.collate_fn) 

    print_dataset_statistics(train_dataset, "train", config)
    print_dataset_statistics(test_dataset, "test", config)

    if pretrained_model_path:
        pretrained_model_path_adapted = f"{pretrained_model_path}_{model_checkpoint.replace('/', '_')}.pth"
    else:
        pretrained_model_path_adapted = None

    final_results, model = train_model(train_dataloader, test_dataloader, model_checkpoint,
                                        model_path=model_save_path, 
                                        early_stopping=True,
                                        pretrained_model_path=pretrained_model_path_adapted,
                                        config=config,
                                        debug=True,
    )
    return final_results, model


def run_cross_validation_for_models(
    sentence_data, 
    kfold_df, 
    config,
    pretrained_model_path_base=None, 
    pretrained_bert_model_path_base=None, 
    model_name="no_name", 
    corpus_name="no_name", 
    add_coreferences=False,
    debug=False
):
    """Cross validation loop for data.
    Train models, aggregate results and collects oof predictions.

    Parameters
    ----------
    sentence_data : list
        List of sentence samples with tokens and relations
    kfold_df : pd.DataFrame
        Fold data for the samples
    config : dict
        Dictionary containing overall parameters and constants
    pretrained_model_path_base : string, optional
        First part of the location of a pre-trained model.
        (Second part is the model checkpoint) , by default None
    pretrained_bert_model_path_base : string, optional
        First part of the location of a pre-trained BERT model.
        (Second part is the model checkpoint) , by default None
    model_name : str, optional
        Name for the model, by default "no_name"
    corpus_name : str, optional
        Name of the training corpus, by default "no_name"
    add_coreferences : str, optional
        Whether to use coreference labels, by default False
    debug : bool, optional
        Whether debug output is provided, by default False

    Returns
    -------
    dict
        Combined results over all folds
    """

    models_results = {}
    for model_checkpoint in config["model_checkpoints"]:

        print(f"Model {model_checkpoint}")

        oof_predictions = []
        oof_results_all = []
        oof_results_filtered = []
        for fold in sorted(kfold_df["fold"].unique()):
            seed_everything()

            print(f"Fold {fold}")

            train_ids = kfold_df.query("fold != @fold")["sentence_id"].to_list()
            test_ids = kfold_df.query("fold == @fold")["sentence_id"].to_list()
            
            # make datasets and loaders
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            train_dataset = CBERTDataset(train_ids, sentence_data, tokenizer, config, only_causal=True, add_coreferences=add_coreferences, use_normalized=config["use_normalized"])
            test_dataset = CBERTDataset(test_ids, sentence_data, tokenizer, config, add_coreferences=add_coreferences, use_normalized=config["use_normalized"])

            if debug:
                print_dataset_statistics(train_dataset, "train", config)
                print_dataset_statistics(test_dataset, "test", config)

            train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_dataset.collate_fn)
            test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=test_dataset.collate_fn) 

            pretrained_model_path = f"{pretrained_model_path_base}_{model_checkpoint.replace('/', '_')}.pth" if pretrained_model_path_base else None
            pretrained_bert_model_path = f"{pretrained_bert_model_path_base}_{model_checkpoint.replace('/', '_')}.pth" if pretrained_bert_model_path_base else None
            _, model = train_model(
                train_dataloader, 
                test_dataloader, 
                model_checkpoint, 
                config=config,
                model_path="model.pth", # dummy path because we don't need it later
                pretrained_model_path=pretrained_model_path,
                pretrained_bert_model_path=pretrained_bert_model_path,
                debug=debug,
            )

            # get oof predictions
            test_results = validate(model, test_dataloader, config, use_labels=True)
            test_results_unlabelled = validate(model, test_dataloader, config, use_labels=False)

            args_result = pd.DataFrame(test_results["arg_results"])
            args_result_unlablled = pd.DataFrame(test_results_unlabelled["arg_results"])
            args_result_unlablled = args_result_unlablled.query("kind == 'Predicted'").copy()
            args_result_unlablled["kind"] = "Predicted without label"
            
            # postprocessing
            args_result_unlablled = postprocessing_predictions(args_result_unlablled, config)

            oof_results_fold = pd.concat([args_result, args_result_unlablled])
            oof_results_fold["fold"] = fold
            oof_predictions.append(oof_results_fold)
            
            # aggregate and store metrics
            # filter out all non-causal relations in GT because prediction can only predict causal and not non-causal
            match_relations_results = calc_relation_metrics(oof_results_fold.query("type != 'None'"), config)
            test_results["match_relations_results"] = match_relations_results
            
            filtered_metrics = filter_metrics_from_validation_results(test_results, config)
            filtered_metrics["Match Relations F1 (strict)"] = match_relations_results["strict"]["f1"]
            filtered_metrics["Match Relations F1 (relaxed)"] = match_relations_results["relaxed"]["f1"]

            if debug:
                display(filtered_metrics)

            oof_results_filtered.append(filtered_metrics)
            oof_results_all.append(test_results)
        
        oof_results_filtered = pd.concat(oof_results_filtered).reset_index(drop=True)
        oof_results_filtered.loc["Combined Results"] = [f"{m:.4f} (+/-{s:.4f})" for m,s in zip(oof_results_filtered.mean(), oof_results_filtered.std())]

        models_results[model_checkpoint] = {
            "oof_predictions": pd.concat(oof_predictions).reset_index(drop=True).to_dict(),
            "oof_results_filtered": oof_results_filtered.to_dict(),
            "oof_results_all": oof_results_all,
        }

    json_path = config["results_directory"] + f"{corpus_name}_predictions_{config['time']}_{model_name}.json"
    with open(json_path, 'w') as f:
        json.dump(models_results, f,  indent=4, cls=NumpyEncoder)

    return models_results


def validate(model, dataloader, config, use_labels=True, debug=False):
    """Validation loop and calculating the metrics for all tasks

    Parameters
    ----------
    model : torch.nn.Module
        Model used for evaluation
    dataloader : torch.DataLoader
        Validation Data
    config : dict
        Dictionary containing overall parameters and constants
    use_labels : bool, optional
        Whether labels are used during forwards pass.
        If True, Tasks are performed with labels of previous Task as input and 
        metrics are computed.
        If False, Tasks use output of previous task, no metrics are computed, 
        by default True
    debug : bool, optional
        Whether debug output is provided, by default False

    Returns
    -------
    dict
        Validation results with predictions, and optionally metrics
    """

    results = defaultdict(list)
    losses = {
        "triggers_loss": 0,
        "combine_triggers_loss": 0,
        "args_loss": 0,
        "type_loss": 0,
        "degree_loss": 0,
    }

    model.eval()
    for batch in dataloader:
        with torch.no_grad():
            batch_results, batch_losses = model(
                    input_ids=batch["input_ids"].to(config["device"]), 
                    attention_masks=batch["attention_masks"].to(config["device"]), 
                    subwords=batch["sample_subwords"].to(config["device"]), 
                    triggers=batch["sample_triggers"].to(config["device"]) if use_labels else None,
                    arguments=batch["sample_relation_arguments"].to(config["device"]) if use_labels else None,
                    types=batch["sample_relation_types"].to(config["device"]) if use_labels else None,
                    degrees=batch["sample_relation_degrees"].to(config["device"]) if use_labels else None,
            )            
            
        # Save losses
        for loss_name, loss in batch_losses.items():
            losses[loss_name] += loss.item() / len(dataloader)

        # Extract results
        for sample in batch_results:
            for pred_name in ["pred_triggers", "pred_combine_triggers", "pred_args", "pred_type", "pred_degree"]:
                results[pred_name].append(sample[pred_name].detach().cpu().numpy()) 

            results["pred_is_causal"].append(sample["pred_is_causal"])
        
    arg_results, trigger_results = get_test_predictions(results, dataloader.dataset, config)

    validation_results = {}
    validation_results["arg_results"] = arg_results.to_dict()
    validation_results["trigger_results"] = trigger_results.to_dict()

    if use_labels: # GT labels provided
        validation_results = aggregate_validation_metrics(arg_results, trigger_results, validation_results, losses, config)
    
    return validation_results


def aggregate_validation_metrics(arg_results, trigger_results, validation_results, losses, config):
    """Compute metrics from all 5 tasks.

    Parameters
    ----------
    arg_results : pd.DataFrame
        Results for argument detection and semantic classification tasks
    trigger_results : pd.DataFrame
        Results for trigger detection and trigger combination tasks
    validation_results : dict
        Dictionary where metrics are stored
    losses : dict
        Losses of the individual tasks
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    dict
        Validation results extended by metrics and losses of individual tasks
    """

    validation_results["detect_trigger_results"] = get_trigger_metrics(
        trigger_results["pred_trigger"].to_list(), 
        trigger_results["true_trigger"].to_list(), 
        losses, config
    )

    validation_results["combine_trigger_results"] = get_combine_trigger_metrics(
        flatten(trigger_results["pred_combine_triggers"].to_list()), 
        flatten(trigger_results["true_combine_triggers"].to_list()), 
        losses
    )

    # combine gt and pred for args
    pred_results = arg_results.query("kind == 'Predicted'").reset_index(drop=True)
    gt_results = arg_results.query("kind == 'Ground Truth'").reset_index(drop=True)
    combined_results = pd.merge(pred_results, gt_results, how="inner", on=["relation_id", "id"], suffixes=("_pred", "_gt"))

    validation_results["detect_args_results"] = get_argument_detection_metrics(combined_results, losses, config)

    validation_results["classify_type_results"] = get_type_metrics(
        gt_results[gt_results["type"] != "None"]["type"].to_list(),
        pred_results[gt_results["type"] != "None"]["type"].to_list(),
        trigger_results, losses, config
    )

    validation_results["classify_degree_results"] = get_degree_metrics(
        gt_results[gt_results["degree"] != "None"]["degree"].to_list(),
        pred_results[gt_results["degree"] != "None"]["degree"].to_list(),
        losses, config
    )

    return validation_results


def annotate_single_sentence(model, sentence, tokenizer, config):
    """Use trained model to annotate a sentence

    Parameters
    ----------
    model : torch.nn.Module
        CBERT model
    sentence : str
        Sentence for causal relation extraction
    tokenizer : Huggingface Tokenizer
        Tokenizer for BERT model in CBERT
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    pd.DataFrame
        Predicted causal relations in sentence
    """

    model.eval()

    tokens = [token.text for token in config["nlp_eng"](sentence) if token.text != "\n"]
    tokenized_inputs = tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=config["max_length"])
    
    # make subword_ids
    word_ids = np.array(tokenized_inputs.word_ids(batch_index=0))
    word_ids[[0, -1]] = -1, len(tokens)+1
    sample_subwords = []
    for i in pd.Series(word_ids).value_counts().sort_index().values:
        sample_subwords += list(range(i))
    
    # CLS and SEP do not count
    sample_subwords[0] = -1
    sample_subwords[-1] = -1

    sample = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

    prediction, _ = model(
        input_ids=sample["input_ids"].to(config["device"]), 
        attention_masks=sample["attention_mask"].to(config["device"]),
        subwords=torch.tensor([sample_subwords]).to(config["device"]),
    )
    prediction = prediction[0]

    results = {}
    for pred_name in ["pred_triggers", "pred_combine_triggers", "pred_args", "pred_type", "pred_degree"]:
        results[pred_name] = prediction[pred_name].detach().cpu().numpy()
    results["pred_is_causal"] = prediction["pred_is_causal"]

    # 0 is CLS and not counting SEP token
    range_slice = slice(1, np.sum(sample["attention_mask"].tolist()) - 1)

    # ---- arguments results ----
    pred_args = results['pred_args']
    pred_type = results['pred_type']
    pred_degree = results['pred_degree']

    arg_results = get_argument_sample_result(tokens, pred_args, pred_type, pred_degree, word_ids.tolist()[1:-1], "Predicted", 0, range_slice, config)
    
    # add dummy results if no causal relation found
    if len(arg_results) == 0:
        arg_results.append(get_dummy_result(tokens, "Predicted", 0))

    return pd.concat(arg_results)