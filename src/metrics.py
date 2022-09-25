import numpy as np
import pandas as pd
from helper_functions import flatten, remove_prefix
from seqeval.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report as classification_report_sk
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef


def compute_metrics(pred_labels, true_labels, config):
    """Compute metrics for sequence label prediction tasks.
    Strict, relaxed, confusions matrices amd classification reports

    Parameters
    ----------
    pred_labels : list
        List of lists of predicted labels for the samples
    true_labels : list
        List of lists of true labels for the samples
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    dict
        Metrics for sequence labeling
    """

    tag_types = sorted(remove_prefix(config["label_dict"].keys()))

    # strict metrics
    results = config["strict_metric"].compute(predictions=pred_labels, references=true_labels, zero_division=False)

    results_mod = {tag_type + "_strict_f1": results[tag_type]["f1"] for tag_type in tag_types if tag_type in results}
    results_mod["overall_precision_strict"] = results["overall_precision"]
    results_mod["overall_recall_strict"] = results["overall_recall"]
    results_mod["overall_f1_strict"] = results["overall_f1"]

    # relaxed metric
    relaxed_results = calculate_relaxed_metric(y_true=true_labels, y_pred=pred_labels)

    for tag_type in tag_types:
        if  tag_type in results:
            results_mod[tag_type + "_relaxed_f1"] = relaxed_results[tag_type]["f1"]

    results_mod["overall_f1_relaxed"] = relaxed_results["overall"]["f1"]
    results_mod["overall_precision_relaxed"] = relaxed_results["overall"]["precision"]
    results_mod["overall_recall_relaxed"] = relaxed_results["overall"]["recall"]

    # classification reports
    results_mod["classification_report_entity"] = classification_report(true_labels, pred_labels, zero_division=False)

    pred_labels_no_prefix = [label[2:] if label!="O" else "O" for label in flatten(pred_labels)]
    true_labels_no_prefix = [label[2:] if label!="O" else "O" for label in flatten(true_labels)]
    tags_no_other = sorted(set(pred_labels_no_prefix + true_labels_no_prefix) - set(["O"]))
    tags_with_other = tags_no_other + ["O"]
    results_mod["classification_report_token"] = classification_report_sk(true_labels_no_prefix, pred_labels_no_prefix, zero_division=False, labels=tags_no_other)

    # token confusion matrix
    results_mod["confusion_matrix_tokens"] = pd.DataFrame(
        confusion_matrix(true_labels_no_prefix, pred_labels_no_prefix, labels=tags_with_other),
        index=tags_with_other, columns=tags_with_other
    ).to_dict()

    return results_mod


def get_named_tuples(labels):
    """Get entities with positions from labels.

    Parameters
    ----------
    labels : list
        List of labels for each token

    Returns
    -------
    list
        List of dicts, dict contains type and position of entity
    """

    entities = []
    current_ent = []
    label_with_B = ""
    for pos, label in enumerate(labels):
        if label == "O":
            if current_ent:
                entities.append((label_with_B, current_ent))
            current_ent = []
            label_with_B = ""
            continue

        current_label_type = label[2:]

        # if new "B" is found, store label
        if label[0] == "B":
            if current_ent:
                entities.append((label_with_B, current_ent))

            current_ent = [pos]
            label_with_B = current_label_type

        # append only if I and same label
        if label[0] == 'I':
            if label_with_B == current_label_type:
                current_ent.append(pos)
            else:
                if current_ent:
                    entities.append((label_with_B, current_ent))
                current_ent = []
                label_with_B = ""

    # append leftover
    if len(current_ent) and label_with_B != "":
        entities.append((label_with_B, current_ent))

    entities = [{"type": typ, "pos": pos} for typ, pos in entities]

    return entities


def calc_precision(tp, fp):
    if (tp + fp) > 0:
        return tp / (tp + fp)
    else:
        return 0


def calc_recall(tp, fn):
    if (tp + fn) > 0:
        return tp / (tp + fn)
    else:
        return 0


def calc_f1(tp, fp, fn):
    if (tp + fn + fp) > 0:
        return tp / (tp + 0.5 * (fp + fn))
    else:
        return 0


def calculate_relaxed_metric(y_true, y_pred):
    """Calculates relaxed metric for prediction, where at least one token needs 
    to be overlapping for tp.

    Parameters
    ----------
    y_true : list
        List of lists containing the ground truth labels
    y_pred : list
        List of lists containing the predicted labels

    Returns
    -------
    dict
        Dictionary with relaxed metrics
    """

    ent_types = (set(remove_prefix(flatten(y_true))) | set(remove_prefix(flatten(y_pred)))) - set(["O"])
    results = {}

    for ent_type in ent_types:
        results[ent_type] = {
            "tp": 0, "fp": 0, "fn": 0,
        }
    
    for y_true_sample, y_pred_sample in zip(y_true, y_pred):
        true_ents = get_named_tuples(y_true_sample)
        pred_ents = get_named_tuples(y_pred_sample)

        for ent_type in ent_types:
            true_ents_type = [ent for ent in true_ents if ent["type"] == ent_type]
            pred_ents_type = [ent for ent in pred_ents if ent["type"] == ent_type]

            true_matched = []
            pred_matched = []

            overlap_mat = np.zeros((len(true_ents_type), len(pred_ents_type)))

            for true_i, true_ent in enumerate(true_ents_type):
                for pred_i, pred_ent in enumerate(pred_ents_type):
                    overlap_mat[true_i, pred_i] = len(set(true_ent["pos"]) & set(pred_ent["pos"])) / len(set(true_ent["pos"]) | set(pred_ent["pos"])) # Jaccard index

            # get matches greedily
            while np.sum(overlap_mat > 0):
                best_score = overlap_mat.max()
                true_match, pred_match = np.where(overlap_mat == best_score)

                for pred_i, true_i in zip(pred_match, true_match):
                    if true_i in true_matched or pred_i in pred_matched: continue

                    true_matched.append(true_i)
                    pred_matched.append(pred_i)

                    overlap_mat[true_i, pred_i] = 0

                overlap_mat[true_match, pred_match] = 0

            results[ent_type]["tp"] += len(true_matched)
            results[ent_type]["fn"] += len(set(np.arange(len(true_ents_type))) - set(true_matched))
            results[ent_type]["fp"] += len(set(np.arange(len(pred_ents_type))) - set(pred_matched))

    for ent_type in ent_types:
        results[ent_type]["precision"] = calc_precision(tp=results[ent_type]["tp"], fp=results[ent_type]["fp"])
        results[ent_type]["recall"] = calc_recall(tp=results[ent_type]["tp"], fn=results[ent_type]["fn"])
        results[ent_type]["f1"] = calc_f1(tp=results[ent_type]["tp"], fp=results[ent_type]["fp"], fn=results[ent_type]["fn"])

    overall_result = {}
    overall_result["tp"] = sum([results[ent_types]["tp"] for ent_types in ent_types])
    overall_result["fn"] = sum([results[ent_types]["fn"] for ent_types in ent_types])
    overall_result["fp"] = sum([results[ent_types]["fp"] for ent_types in ent_types])

    overall_result["precision"] = calc_precision(tp=overall_result["tp"], fp=overall_result["fp"])
    overall_result["recall"] = calc_recall(tp=overall_result["tp"], fn=overall_result["fn"])
    overall_result["f1"] = calc_f1(tp=overall_result["tp"], fp=overall_result["fp"], fn=overall_result["fn"])

    results["overall"] = overall_result
    return results


def calc_relation_metrics(oof_results, config, strict_thresh=0.9):
    """Matches the relations and returns strict and relaxed metrics.
    - Relaxed: only one tokens has to overlap for a match
    - Strict: f1 score of over strict_threshold needed for a match

    Parameters
    ----------
    oof_results : pd.DataFrame
        Contains the predicted out-of-fold results
    config : dict
        Dictionary containing overall parameters and constants
    strict_thresh : float, optional
        Threshold of overlap for a strict match to occur, by default 0.9

    Returns
    -------
    dict
        Contains strict and relaxed results of matched relations
    """
    all_matches = []
    for i in oof_results["id"].unique():
        all_matches += get_matches_from_sample(oof_results.query("id == @i"), config)

    return get_metrics_from_matches(all_matches, strict_thresh)


def get_matches_from_sample(sample_data, config):
    """Helper function to retrive matches from Ground Truth and Predicted 
    sample.

    Parameters
    ----------
    sample_data : pd.DataFrame
        Contains ground truth and predictions for a sample
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List of matches between predicted and ground truth relation with score
    """
    pred_data = sample_data.query("kind == 'Predicted without label'")
    gt_data = sample_data.query("kind == 'Ground Truth'")

    return get_matches(pred_data, gt_data, config)
    

def get_matches(pred_data, gt_data, config, only_trigger=False):
    """Calculate matches from predicted and ground truth relations.
    Greedy algorithm, relations with highest F1 score on a token-level are 
    matched iteratively. After a relation is matched, it cannot be matched 
    again.

    Parameters
    ----------
    pred_data : pd.DataFrame
        Contains predicted relations for a sample
    gt_data : pd.DataFrame
        Contains ground truth relations for a sample
    config : dict
        Dictionary containing overall parameters and constants
    only_trigger : bool
        Whether only Trigger tokens should be used for calculating the f1 score

    Returns
    -------
    list
        List of matches between predicted and ground truth relation with score
    """
    
    # no causal relation for GT and pred
    if len(pred_data) == 1 and len(gt_data) == 1 and \
        np.all(pred_data.iloc[0]["labels"] == "O") and \
        np.all(gt_data.iloc[0]["labels"] == "O"):
        return [{ "gt_rel_id": 0, "pred_rel_id": 0, "score": -1}]

    # Calculate overlap comparison scores using f1
    if only_trigger:
        prefix_no_other = ["Trigger"]
    else:
        prefix_no_other = list(set(remove_prefix(config["label_list"])) - set("O") | set(["Trigger"]))
    
    pred_relation_ids = pred_data["relation_id"].unique()
    gt_relation_ids = gt_data["relation_id"].unique()
    comparison_scores = []
    for gt_rel_id in gt_relation_ids:
        gt_labels = remove_prefix(gt_data.query("relation_id == @gt_rel_id").iloc[0]["labels"])

        for pred_rel_id in pred_relation_ids:
            pred_labels = remove_prefix(pred_data.query("relation_id == @pred_rel_id").iloc[0]["labels"])

            comparison_scores.append({
                "gt_rel_id": gt_rel_id,
                "pred_rel_id": pred_rel_id,
                "score": f1_score(gt_labels, pred_labels, labels=prefix_no_other, average="micro", zero_division=False)
            })

    # sort according to f1
    comparison_scores = pd.DataFrame(comparison_scores)
    if len(comparison_scores):
        comparison_scores = comparison_scores.sort_values("score", ascending=False)

    # greedily match and remove matched relations
    matches = []
    while len(comparison_scores):
        # get pair with highest score
        current_best_match = comparison_scores.iloc[0]
        matches.append({
            "gt_rel_id": current_best_match["gt_rel_id"],
            "pred_rel_id": current_best_match["pred_rel_id"],
            "score": current_best_match["score"],
        })

        # remove choosen prediction and gt df
        comparison_scores = comparison_scores[comparison_scores["pred_rel_id"] != current_best_match["pred_rel_id"]]
        comparison_scores = comparison_scores[comparison_scores["gt_rel_id"] != current_best_match["gt_rel_id"]]

    # add leftover gt
    for gt_rel_id in set(gt_relation_ids) - {m["gt_rel_id"] for m in matches}:
        matches.append({
            "gt_rel_id": gt_rel_id,
            "pred_rel_id": -1,
            "score": 0,
        })

    # add leftover pred
    for pred_rel_id in set(pred_relation_ids) - {m["pred_rel_id"] for m in matches}:
        matches.append({
            "gt_rel_id": -1,
            "pred_rel_id": pred_rel_id,
            "score": 0,
        })

    return matches


def get_metrics_from_matches(matches, strict_thresh=0.9):
    """Calculate strict and relaxed metrics from relation matches.

    Parameters
    ----------
    matches : list
        List of matches between predicted and ground truth relation with score
    strict_thresh : float, optional
        Threshold of overlap for a strict match to occur, by default 0.9

    Returns
    -------
    dict
        Contains strict and relaxed results of matched relations
    """

    results = {
        "strict": {"tp": 0, "fp": 0, "fn": 0},
        "relaxed": {"tp": 0, "fp": 0, "fn": 0},
    }

    for match in matches:
        if match["gt_rel_id"] == -1: 
            # no gt left for pred -> fp
            results["strict"]["fp"] += 1
            results["relaxed"]["fp"] += 1

        elif match["pred_rel_id"] == -1: 
            # no pred left for gt -> fn
            results["strict"]["fn"] += 1
            results["relaxed"]["fn"] += 1

        else:
            if match["score"] == 0:
                # both pred and gt exits but no match
                results["relaxed"]["fp"] += 1
                results["relaxed"]["fn"] += 1

                results["strict"]["fp"] += 1
                results["strict"]["fn"] += 1

            elif strict_thresh > match["score"] > 0:
                # some match -> relaxed tp
                results["relaxed"]["tp"] += 1

                # some match -> strict no match -> fp + fn for pred and gt
                results["strict"]["fp"] += 1
                results["strict"]["fn"] += 1

            elif match["score"] >= strict_thresh:
                # full match
                results["relaxed"]["tp"] += 1
                results["strict"]["tp"] += 1

    results["strict"]["precision"] = calc_precision(tp=results["strict"]["tp"], fp=results["strict"]["fp"])
    results["strict"]["recall"] = calc_recall(tp=results["strict"]["tp"], fn=results["strict"]["fn"])
    results["strict"]["f1"] = calc_f1(tp=results["strict"]["tp"], fp=results["strict"]["fp"], fn=results["strict"]["fn"])

    results["relaxed"]["precision"] = calc_precision(tp=results["relaxed"]["tp"], fp=results["relaxed"]["fp"])
    results["relaxed"]["recall"] = calc_recall(tp=results["relaxed"]["tp"], fn=results["relaxed"]["fn"])
    results["relaxed"]["f1"] = calc_f1(tp=results["relaxed"]["tp"], fp=results["relaxed"]["fp"], fn=results["relaxed"]["fn"])

    return results


def get_trigger_metrics(pred, true, losses, config):
    detect_trigger_results = compute_metrics(
            pred,
            true,
            config
    )
    detect_trigger_results["triggers_loss"] = losses["triggers_loss"]     
    return detect_trigger_results
    

def get_combine_trigger_metrics(pred, true, losses):
    combine_trigger_results = {
        "f1_macro": f1_score(true, pred, zero_division=False, average="macro"),
        "accuracy": accuracy_score(true, pred),
        "MCC": matthews_corrcoef(true, pred),
        "% True": np.mean(true),
        "combine_trigger_loss": losses["combine_triggers_loss"],
    }
    return combine_trigger_results


def get_argument_detection_metrics(combined_results, losses, config):
    detect_args_results  = compute_metrics(
        [[i if i[2:] != "Trigger" else "O" for i in l] for l in combined_results["labels_pred"].to_list()],
        [[i if i[2:] != "Trigger" else "O" for i in l] for l in combined_results["labels_gt"].to_list()],
        config
    )
    detect_args_results["args_loss"] = losses["args_loss"]
    return detect_args_results


def get_type_metrics(pred, true, trigger_results, losses, config):

    type_report = classification_report_sk(true, pred, zero_division=False, output_dict=True)
    types_no_None = config["type_list"][:3]
    confusion_matrix_type = pd.DataFrame(
            confusion_matrix(true, pred, labels=types_no_None),
            index=types_no_None, columns=types_no_None
    ).to_dict()

    true_is_causal = trigger_results["true_is_causal"].to_list()
    pred_is_causal = trigger_results["pred_is_causal"].to_list()

    return {
        "accuracy": accuracy_score(true, pred),
        "MCC": matthews_corrcoef(true, pred),
        "f1_macro": f1_score(true, pred, zero_division=False, average="macro"),
        "type_loss": losses["type_loss"],
        "purpose_f1": type_report["Purpose"]["f1-score"],
        "motivation_f1": type_report["Motivation"]["f1-score"],
        "consequence_f1": type_report["Consequence"]["f1-score"],
        "is_causal_accuracy": accuracy_score(true_is_causal, pred_is_causal),
        "is_causal_f1": f1_score(true_is_causal, pred_is_causal),
        "% causal": sum(true_is_causal) / len(true_is_causal),
        "confusion_matrix": confusion_matrix_type,
    }


def get_degree_metrics(pred, true, losses, config):
    degree_report = classification_report_sk(true, pred, zero_division=False, output_dict=True)
    degrees_no_None = config["degree_list"][:2]
    confusion_matrix_degree = pd.DataFrame(
            confusion_matrix(true, pred, labels=degrees_no_None),
            index=degrees_no_None, columns=degrees_no_None
    ).to_dict()

    return {
        "accuracy": accuracy_score(true, pred),
        "MCC": matthews_corrcoef(true, pred),
        "f1_macro": f1_score(true, pred, zero_division=False, average="macro"),
        "facilitate_f1": degree_report["Facilitate"]["f1-score"],
        "inhibit_f1": degree_report["Inhibit"]["f1-score"],
        "degree_loss": losses["degree_loss"],
        "% facilitate": true.count("Facilitate") / len(true),
        "confusion_matrix": confusion_matrix_degree,
    }
