import numpy as np
import pandas as pd
from metrics import get_matches_from_sample
from spacy import displacy


def get_indices_for_label(labels, config):
    """Extract indicides from the labels for each causal argument

    Parameters
    ----------
    labels : list
        Contains labels in BIO format
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    np.array
        Array containing arguments with corresponding positions
    """

    sentence_arguments = []
    for causal_arg in config["causal_arguments"]:

        argument = []
        for i, label in enumerate(labels):
            if label == f"B-{causal_arg}":
                if len(argument):
                    sentence_arguments.append((causal_arg, argument))
                argument = [i]

            elif label == f"I-{causal_arg}":
                argument.append(i)

            elif len(argument):
                sentence_arguments.append((causal_arg, argument))
                argument = []
        
        if len(argument):
            sentence_arguments.append((causal_arg, argument))

    if len(sentence_arguments) == 0:
        return []

    return pd.DataFrame(sentence_arguments).sort_values(1).values


def visualize_annotated_labels(tokens, labels, config):
    """Display tokens with labels using displacy and return the html render

    Parameters
    ----------
    tokens : list
        Tokens in the sample
    labels : list
        BIO labels in the sample
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    str
        Html string of displacy render
    """

    text_split = tokens
    text = " ".join(text_split)
    span_starts = np.array([0] + list(map(len, text_split))).cumsum()

    ents = []
    for i, (causal_arg, label) in enumerate(get_indices_for_label(labels, config)):

        ents.append({
            'start': span_starts[label[0]] + label[0], 
            'end': span_starts[label[-1]+1] + label[-1], 
            'label': causal_arg
        })

    doc = {
        "text": text,
        "ents": ents,
    }

    return displacy.render(doc, style="ent", manual=True, jupyter=False, options=config["entity_options"])


def get_sample_with_layout_string(sample, kind, config):
    """Construct html string for a single sample.

    Parameters
    ----------
    sample : pd.DataFrame
        Contains the tokens, labels, type and degree of sample
    kind : str
        Kind of data (usually "Ground Truth" or "Predicted")
   config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    str 
        Html string of sample
    """
    sample = sample.iloc[0]

    string = ""
    string += f"<h4>{kind}</h4>"
    string += f"Causal Type: <u>{sample['type']}</u>, Degree: <u>{sample['degree']}</u>"
    string += visualize_annotated_labels(sample["tokens"], sample["labels"], config)
    return string


def get_empty_with_layout_string(kind):
    """Construct html string empty sample.

    Parameters
    ----------
    kind : str
        Kind of data (usually "Ground Truth" or "Predicted")

    Returns
    -------
    str 
        Html string of empty sample
    """
    string = ""
    string += f"<h4>{kind}</h4>"
    string += f"<b><i>No match found</i></b>"
    return string


def visualize_id_html(id, oof_results, config):
    """Generate visualization html of the OOF results for a sample given by id.

    Parameters
    ----------
    id : int
        Id to visualize
    oof_results : pd.DataFrame
        OOF results of causal relations
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    str
        Html string for visualizing the sample
    """
    sample_data = oof_results.query("id == @id")
    matches = get_matches_from_sample(sample_data, config)

    string = ""
    string += f"<h2>Sample {id}</h2>"
    for i, (match) in enumerate(matches):
        string += f"<h3>Relation <u>{i}</u> with score: <u>{match['score']:.3f}</u></h3>"

        gt_rel_id = match["gt_rel_id"]
        if gt_rel_id != -1:
            gt_data = sample_data.query("kind == 'Ground Truth' & relation_id == @gt_rel_id")
            string += get_sample_with_layout_string(gt_data, kind="Ground Truth", config=config)
        else:
            string += get_empty_with_layout_string("Ground Truth")

        pred_rel_id = match["pred_rel_id"]
        if pred_rel_id != -1:
            pred_data = sample_data.query("kind == 'Predicted without label' & relation_id == @pred_rel_id")
            string += get_sample_with_layout_string(pred_data, kind="Predicted", config=config)
        else:
            string += get_empty_with_layout_string("Predicted")

    string += f"<hr>"
    return string
