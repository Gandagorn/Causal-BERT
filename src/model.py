import networkx as nx
import numpy as np
import pandas as pd
import torch
from helper_functions import get_trigger_comb_input, make_trigger_groups
from transformers import AutoModel


class TRIGGER_CBERT(torch.nn.Module):
    """
    Model to detect triggers in a sequence.
    """

    def __init__(self, config):
        super(TRIGGER_CBERT, self).__init__()

        self.config = config

        self.trigger_classifier = torch.nn.Linear(config["bert_embedding_size"], 2)
        self.loss_function = torch.nn.CrossEntropyLoss()


    def forward(self, bert_embeddings, labels=None):

        pred_triggers = self.trigger_classifier(bert_embeddings)

        if labels is None:
            loss = torch.tensor(0.).to(self.config["device"])
        else:
            loss = self.loss_function(pred_triggers.permute(0, 2, 1), labels)
        
        return pred_triggers, loss


class COMB_TRIGGER_CBERT(torch.nn.Module):
    """
    Model to predict whether two Triggers are part of the same trigger group.
    """

    def __init__(self, config):
        super(COMB_TRIGGER_CBERT, self).__init__()

        self.config = config

        self.linear = torch.nn.Linear(config["bert_embedding_size"]*2, config["bert_embedding_size"])
        self.gelu = torch.nn.GELU()
        self.layer_norm = torch.nn.LayerNorm(768, eps=1e-12)
        self.linear2 = torch.nn.Linear(config["bert_embedding_size"], 1)
        self.loss_function = torch.nn.BCELoss()

    def forward(self, embeddings, labels=None):

        pred_combination = self.linear(embeddings)
        pred_combination = self.gelu(pred_combination)
        pred_combination = self.layer_norm(pred_combination)
        pred_combination = self.linear2(pred_combination)

        pred_combination = torch.sigmoid(pred_combination).view(len(embeddings))

        if labels is None or len(labels) == 0:
            loss = torch.tensor(0.).to(self.config["device"])
        else:
            loss = self.loss_function(pred_combination, labels)

        return pred_combination, loss


class DETECT_ARGS_CBERT(torch.nn.Module):
    """
    Model to predict the causal arguments (except trigger).
    """

    def __init__(self, config):
        super(DETECT_ARGS_CBERT, self).__init__()

        self.config = config

        self.linear = torch.nn.Linear(config["bert_embedding_size"]*2, config["bert_embedding_size"])
        self.linear2 = torch.nn.Linear(config["bert_embedding_size"], 256)
        self.linear3 = torch.nn.Linear(256, len(config["label_list"])-1) # B-Trigger are not predicted
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):

        pred_args = self.linear(embeddings)
        pred_args = torch.nn.functional.relu(pred_args)
        pred_args = self.linear2(pred_args)
        pred_args = torch.nn.functional.relu(pred_args)
        pred_args = self.linear3(pred_args)

        if labels is None:
            loss = torch.tensor(0.).to(self.config["device"])
        else:
            loss = self.loss_function(pred_args, labels)

        return pred_args, loss


class CLASSIFY_TYPE_CBERT(torch.nn.Module):
    """
    Model to predict the causal type of a trigger group embedding.
    """

    def __init__(self, config):
        super(CLASSIFY_TYPE_CBERT, self).__init__()

        self.config = config

        self.linear = torch.nn.Linear(config["bert_embedding_size"], 3)
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, embeddings, labels=None):

        pred_type = self.linear(embeddings)

        if labels is None:
            loss = torch.tensor(0.).to(self.config["device"])
        else:
            loss = self.loss_function(pred_type, labels)

        return pred_type, loss


class CLASSIFY_DEGREE_CBERT(torch.nn.Module):
    """
    Model to predict the causal degree of a trigger group embedding.
    """

    def __init__(self, config):
        super(CLASSIFY_DEGREE_CBERT, self).__init__()

        self.config = config

        self.linear = torch.nn.Linear(config["bert_embedding_size"], 1)
        self.loss_function = torch.nn.BCELoss()

    def forward(self, embeddings, labels=None):

        pred_degree = self.linear(embeddings)
        pred_degree = torch.sigmoid(pred_degree).flatten()

        if labels is None:
            classify_degree_loss = torch.tensor(0.).to(self.config["device"])
        else:
            classify_degree_loss = self.loss_function(pred_degree, labels.float())

        return pred_degree, classify_degree_loss


class CONTEXT_ATTENTION(torch.nn.Module):
    """
    Simple attention model based on 
    https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
    """

    def __init__(self, config):
        super(CONTEXT_ATTENTION, self).__init__()

        self.linear = torch.nn.Linear(config["bert_embedding_size"], 100)
        self.causal_context = torch.nn.Parameter(torch.rand(100, requires_grad=True))


    def forward(self, trigger_embeddings):

        u = self.linear(trigger_embeddings)
        u = torch.nn.functional.relu(u)
        alpha = torch.nn.functional.softmax(u @ self.causal_context, dim=0)

        return alpha.view(-1,1)


class CBERT(torch.nn.Module):
    """
    Main Causal-BERT model, contains all sub-models for the individual tasks 
    and handels the loss and output computations.
    All tasks are done iteratively.
    If labels are provided, labels of previous tasks are used in subsequent tasks.
    If no labels are provided, output of previous tasks is used in subsequent tasks.
    """

    def __init__(self, model_checkpoint, config):
        super(CBERT, self).__init__()

        self.config = config

        self.bert = AutoModel.from_pretrained(model_checkpoint)

        self.dropout = torch.nn.Dropout(config["bert_dropout"])

        self.trigger_model = TRIGGER_CBERT(config)
        self.detect_args_model = DETECT_ARGS_CBERT(config)
        self.combine_trigger_model = COMB_TRIGGER_CBERT(config)
        self.classify_type_model = CLASSIFY_TYPE_CBERT(config)
        self.classify_degree_model = CLASSIFY_DEGREE_CBERT(config)

        self.context_attention = CONTEXT_ATTENTION(config)


    def predict_func(self, pred_triggers, bert_embeddings, subwords, losses):
        """Predict function for tasks 3-5, used if no ground truth is provided.
        The output of the previous task is used in the next task.

        Parameters
        ----------
        pred_triggers : torch.tensor
            Tensor of trigger predictions from task 2
        bert_embeddings : torch.tensor
            BERt embeddings from task 1, used as features in all other tasks
        subwords : torch.tensor
            Tensor where subwords of tokens are numerated, 0 values are start
        losses : dict
            Dictionary with losses for all tasks (not used)

        Returns
        -------
        list, dict
            List of results for all samples in batch and dictionary of losses
        """

        # for placeholder results
        dummy_tensor = torch.tensor([], dtype=int, device=self.config["device"])

        batch_results = []
        for sample_id in range(pred_triggers.shape[0]):

            sample_pred_triggers = pred_triggers[sample_id]
            sample_embeddings = bert_embeddings[sample_id]
            sample_subwords = subwords[sample_id]

            # Get Trigger results
            single_results = {}
            sample_pred_triggers = sample_pred_triggers.argmax(1)

            single_results["pred_triggers"] = sample_pred_triggers
            trigger_ind = torch.where( (sample_pred_triggers==self.config["label_dict"]["B-Trigger"]) & (sample_subwords==0))[0].unique()

            # If more than 20 Triggers are found, take 20 randomly
            if trigger_ind.shape[0] > 20:
                trigger_ind = torch.randperm(trigger_ind.shape[0])[:20]
            trigger_ind, _ = torch.sort(trigger_ind)

            # Find Trigger combinations
            trigger_combs =  torch.combinations(trigger_ind, r=2)
            trigger_combs_embeddings = sample_embeddings[trigger_combs, :].view(len(trigger_combs), self.config["bert_embedding_size"]*2)
            pred_combine_triggers, _ = self.combine_trigger_model(trigger_combs_embeddings)
            single_results["pred_combine_triggers"] = pred_combine_triggers

            # Find Trigger groups in the graph (in the form of maximal cliques)
            G = nx.Graph()
            G.add_nodes_from(trigger_ind.detach().cpu().numpy())
            G.add_edges_from(trigger_combs[pred_combine_triggers > 0.5].detach().cpu().numpy())
            trigger_groups = [torch.tensor(c) for c in nx.find_cliques(G)]

            # Predict the additional causal arguments and semantic types
            pred_args_sample = []
            pred_type_sample = []
            pred_degree_sample = []
            for trigger_group in trigger_groups: 
                
                # Construct Trigger group embedding
                context_weights = self.context_attention(sample_embeddings[trigger_group, :])
                relation_context_embeddings = torch.sum(context_weights * sample_embeddings[trigger_group, :], dim=0)

                # Combine Trigger group embedding with token embeddings
                combined_embeddings = torch.cat([sample_embeddings, relation_context_embeddings.repeat(sample_embeddings.shape[0], 1)], 1)

                # Predict Arguments
                pred_args, _ = self.detect_args_model(combined_embeddings)
                pred_args = pred_args.argmax(axis=1)

                # Change predicted arguments so Triggers are included
                pred_args = pred_args + 1 # reshift all labels by 1 to account for B-Trigger
                pred_args[trigger_group] = sample_pred_triggers[trigger_group]
                pred_args_sample.append(pred_args)
                
                # Predict Type
                pred_type, _ = self.classify_type_model(relation_context_embeddings.view(1, -1))
                pred_type = pred_type.argmax(axis=1)
                pred_type_sample.append(pred_type[0])

                # Predict Degree
                pred_degree, _ = self.classify_degree_model(relation_context_embeddings.view(1, -1))
                pred_degree = (pred_degree > 0.5).int()
                pred_degree_sample.append(pred_degree[0])

            pred_causal = len(trigger_groups) > 0 
            single_results["pred_args"] = torch.stack(pred_args_sample) if len(pred_args_sample) else dummy_tensor
            single_results["pred_type"] = torch.stack(pred_type_sample) if pred_causal else dummy_tensor
            single_results["pred_degree"] = torch.stack(pred_degree_sample) if pred_causal else dummy_tensor
            single_results["pred_is_causal"] = pred_causal

            batch_results.append(single_results)

        return batch_results, losses


    def training_func(self, pred_triggers, bert_embeddings, subwords, arguments, types, degrees, losses):
        """Training function for tasks 3-5, used if ground truth is provided.
        The ground truth labels of a task are used as input for the next task.

        Parameters
        ----------
        pred_triggers : torch.tensor
            Tensor of trigger predictions from task 2
        bert_embeddings : torch.tensor
            BERt embeddings from task 1, used as features in all other tasks
        subwords : torch.tensor
            Tensor where subwords of tokens are numerated, 0 values are start
        arguments : torch.tensor
            Argument labels of relations (Len 16 as max. 16 relations possible)
        types : torch.tensor
            Type labels of relations (Len 16 as max. 16 relations possible)
        degrees : torch.tensor
            Degree labels of relations (Len 16 as max. 16 relations possible)
        losses : dict
            Dictionary with losses for all tasks (not used)

        Returns
        -------
        list, dict
            List of results for all samples in batch and dictionary of losses
        """

        # for placeholder results
        dummy_tensor = torch.tensor([], dtype=int, device=self.config["device"]) 

        batch_results = []
        for sample_id in range(len(arguments)):

            sample_relation_arguments = arguments[sample_id]
            sample_relation_types = types[sample_id]
            sample_relation_degrees = degrees[sample_id]
            sample_embeddings = bert_embeddings[sample_id]
            sample_subwords = subwords[sample_id]

            # get Trigger results
            sample_results = {}
            sample_pred_triggers = pred_triggers[sample_id].argmax(axis=1)

            sample_results["pred_triggers"] = sample_pred_triggers

            # for None-type no combination and argument detection is done (as no triggers are available)
            is_causal = (sample_relation_types[0] != self.config["type_dict"]["None"]).item()
            if is_causal:
                
                # Get GT Trigger groups
                trigger_groups = make_trigger_groups(sample_relation_arguments, sample_subwords, self.config)
                trigger_combs, trigger_combs_labels = get_trigger_comb_input(trigger_groups)

                # Make Trigger combinations
                trigger_combs_labels = torch.tensor(trigger_combs_labels, device=self.config["device"])
                trigger_combs_embeddings = sample_embeddings[trigger_combs, :].view(len(trigger_combs), self.config["bert_embedding_size"]*2)
                
                pred_combine_triggers, combine_triggers_loss = self.combine_trigger_model(trigger_combs_embeddings, trigger_combs_labels)
                losses["combine_triggers_loss"] += combine_triggers_loss
                sample_results["pred_combine_triggers"] = pred_combine_triggers > 0.5

                # do classification with embeddings from triggers (with GT data)
                # for each relation, predict causal arguments and type
                pred_args_sample = []
                pred_type_sample = []
                pred_degree_sample = []
                for rel_id, trigger_group in enumerate(trigger_groups): 

                    relation_arguments = torch.clone(sample_relation_arguments[rel_id])

                    # construct Trigger group embedding
                    context_weights = self.context_attention(sample_embeddings[trigger_group, :])
                    relation_context_embeddings = torch.sum(context_weights * sample_embeddings[trigger_group, :], dim=0)
                    combined_embeddings = torch.cat([sample_embeddings, relation_context_embeddings.repeat(sample_embeddings.shape[0], 1)], 1)

                    # change labels so that triggers are not additionally predicted again
                    b_triggers = torch.where(relation_arguments==self.config["label_dict"]["B-Trigger"])[0]
                    relation_arguments[b_triggers] = -100
                    valid_arguments = relation_arguments != -100
                    relation_arguments[valid_arguments] = relation_arguments[valid_arguments] - 1 # shift all labels by 1 as B Trigger is 0

                    # predict arguments
                    pred_args, rel_arg_loss = self.detect_args_model(combined_embeddings, relation_arguments)
                    losses["args_loss"] += rel_arg_loss
                    pred_args = pred_args.argmax(axis=1)
                    
                    # change predicted arguments back
                    pred_args[valid_arguments] = pred_args[valid_arguments] + 1 # reshift all labels by 1 as B Trigger is 0
                    pred_args[b_triggers] = self.config["label_dict"]["B-Trigger"] # include triggers
                    pred_args_sample.append(pred_args)

                    # classify type
                    pred_type, rel_type_loss = self.classify_type_model(relation_context_embeddings.view(1, -1), sample_relation_types[rel_id].view(1))
                    losses["type_loss"] += rel_type_loss
                    pred_type = pred_type.argmax(axis=1)
                    pred_type_sample.append(pred_type[0])

                    # classify degree
                    pred_degree, rel_degree_loss = self.classify_degree_model(relation_context_embeddings.view(1, -1), sample_relation_degrees[rel_id].view(1))
                    losses["degree_loss"] += rel_degree_loss
                    pred_degree = (pred_degree > 0.5).int()
                    pred_degree_sample.append(pred_degree[0])
            else:
                # no causal relation in sample -> insert empty dummy
                sample_results["pred_combine_triggers"] = dummy_tensor

            # whether model predicts causal or non-causal
            pred_causal = torch.sum(sample_pred_triggers==0).item() > 0

            # combine relation results to sample
            sample_results["pred_args"] = torch.stack(pred_args_sample) if is_causal else dummy_tensor
            sample_results["pred_type"] = torch.stack(pred_type_sample) if is_causal else dummy_tensor
            sample_results["pred_degree"] = torch.stack(pred_degree_sample) if is_causal else dummy_tensor
            sample_results["pred_is_causal"] = pred_causal

            batch_results.append(sample_results)

        return batch_results, losses


    def forward(self, input_ids, attention_masks, subwords, triggers=None, arguments=None, types=None, degrees=None):
        """Tasks 1 and 2 are computed the same.
        If ground truth labels are provided, tasks 3-5 are computed using the
        ground truth labels of the previous task as input in the next task. 
        If no labels are provided, the output of the previous task is used as
        input for the next task.

        Parameters
        ----------
        input_ids : torch.tensor
            BERT input_ids of batch
        attention_masks : torch.tensor
            BERT attention masks for input_ids
        subwords : torch.tensor
            Tensor where subwords of tokens are numerated, 0 values are start
        triggers : torch.tensor, optional
            Trigger labels of sample, does not distinguish between relation, 
            by default None
        arguments : torch.tensor, optional
            Argument labels of relations (Len 16 as max. 16 relations possible), 
            by default None
        types : torch.tensor, optional
            Type labels of relations (Len 16 as max. 16 relations possible), 
            by default None
        degrees : torch.tensor, optional
            Degree labels of relations (Len 16 as max. 16 relations possible), 
            by default None

        Returns
        -------
        list, dict
            List of results for all samples in batch and dictionary of losses
        """
        
        # construct BERT embeddings
        bert_embeddings = self.bert(input_ids, attention_masks)["last_hidden_state"]
        bert_embeddings = self.dropout(bert_embeddings)

        # predict all Triggers in relations
        pred_triggers, triggers_loss = self.trigger_model(
            bert_embeddings=bert_embeddings, 
            labels=triggers,
        )

        losses = {
            "triggers_loss": triggers_loss,
            "combine_triggers_loss": torch.tensor(0., device=self.config["device"]),
            "args_loss": torch.tensor(0., device=self.config["device"]),
            "type_loss": torch.tensor(0., device=self.config["device"]),
            "degree_loss": torch.tensor(0., device=self.config["device"]),
        }

        if triggers is None:
            return self.predict_func(
                pred_triggers, 
                bert_embeddings, 
                subwords, 
                losses
            )
        else:
            return self.training_func(
                pred_triggers, 
                bert_embeddings, 
                subwords, 
                arguments, 
                types, 
                degrees, 
                losses
            )