import functools
import itertools
import json
import os
import random
import re
from collections import defaultdict
import traceback

import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from pandas_ods_reader import read_ods
from sklearn.metrics import f1_score
from tqdm import tqdm
from tqdm.auto import tqdm

def seed_everything(seed=42):
    """Ensure reproducibility of model training

    Parameters
    ----------
    seed : int, optional
        Seed to set, by default 42
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def flatten(l):
    """Reduces the dimension of a list by 1.

    Parameters
    ----------
    l : list
        List with more than one dimension

    Returns
    -------
    list
        Flattened list
    """
    return [item for sublist in l for item in sublist]


def remove_prefix(labels):
    """Remove prefixes "B-"/"I-" from labels and only keep argument information

    Parameters
    ----------
    labels : list
        List of labels of the form {B|I}-{causal argument} or O

    Returns
    -------
    list
        List of labels of the form {causal argument | O}
    """
    return [l if l == "O" else l[2:] for l in labels]


def parse_brat_annotations(ann_path, config):
    """Helper function for parsing of brat .ann file. 
    Split results into entities, degrees, relations and coreferences. 
    Does not parse non-causal or means arguments.

    Parameters
    ----------
    ann_path : str
        Location of the .ann file to parse
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    dict
        Dictionary containing entities, degrees, relations and coreference
        information
    """

    with open(ann_path, "r") as f:
        lines =  f.read().strip().split("\n")

    entities = {}
    degrees = {}
    relations = {}
    coreferences = {}
    for line in lines:
        
        ann_type = line[0]
        line = line.split("\t")
        if ann_type == "T":
            ent_name, argument, text = line
            spans = argument.split(";")
            ent_spans = []

            for i, span in enumerate(spans):
                span = span.split(" ")

                if i == 0:
                    span = span[1:] # remove causal type

                start, end = span
                ent_spans.append([int(start), int(end)])

            entities[ent_name] = {"spans": ent_spans, "text": text}

        elif ann_type == "E":
            rel_name = line[0]
            rel_dict = defaultdict(list)     

            for arg in line[1].split():
                arg_type, ent_name = arg.split(":")

                # remove possible digit at end (eg. Cause2, Effect3)
                arg_type = re.sub(r'[0-9]+', '', arg_type)

                if arg_type in ["Consequence", "Purpose", "Motivation", "NonCausal"]:
                    rel_dict["rel_type"].append(arg_type)
                    rel_dict["Trigger"].append(ent_name)
                elif arg_type in config["causal_arguments"]:
                    rel_dict[arg_type].append(ent_name)
                else:
                    # print(line) # NonCausal or Means relations, not used
                    pass

            relations[rel_name] = rel_dict

        elif ann_type == "A":
            att_name = line[0]

            if len(line[1].split()) == 2:
                # print("Other relations (ignored)", line)
                continue

            _, rel_name, degree = line[1].split()
            degrees[att_name] = {"rel_name": rel_name, "degree": degree}

        # Coreferences
        elif ann_type == "R":
            ref_name, ref_arguments = line[:2]
            ref_arguments = ref_arguments.split(" ")[1:] # remove "Coref"

            ref1 = ref_arguments[0].split(":")[1]
            ref2 = ref_arguments[1].split(":")[1]
            coreferences[ref_name] = {"ref1": ref1, "ref2": ref2}

    entities = pd.DataFrame(entities).T
    degrees = pd.DataFrame(degrees).T
    relations = pd.DataFrame(relations).T
    coreferences = pd.DataFrame(coreferences, index=["ref_name", "ref1", "ref2"]).T

    entities["ent_name"] = entities.index
    degrees["deg_name"] = degrees.index
    relations["rel_name"] = relations.index
    coreferences["ref_name"] = coreferences.index

    return entities, degrees, relations, coreferences


def get_brat_data(config, ann_path=None, txt_path=None, nlp=None, sentence_split_pattern="\n"):
    """Extracts information from brat and text file.
    Parses .ann file, removes NonCausal, combines relations with multiple 
    Triggers, loads tokens and splits constructs smaller chunks of the data.

    Parameters
    ----------
    config : dict
        Dictionary containing overall parameters and constants
    ann_path : str, optional
        Location of the .ann file to parse, by default None
    txt_path : str, optional
        Location of the .txt file of the text, by default None
    nlp : spacy.lang, optional
        Spacy instance of a language, by default None
    sentence_split_pattern : str, optional
        Pattern to chunk the text. If "", spacy sentence splitting is used, 
        by default "\n"

    Returns
    -------
    dict
        Dictionary containing entities, degrees, relations and coreference
        information
    """

    if ann_path is None:
        ann_path = config["ann_path"]
    
    if txt_path is None:
        txt_path = config["txt_path"]

    if nlp is None:
        nlp = config["nlp_ger"]

    entities, degrees, relations, coreferences = parse_brat_annotations(ann_path, config)

    # Remove NonCausal relations
    relations = relations[relations["rel_type"].str[0] != "NonCausal"] 

    # Combine relations with multiple Triggers
    relations["rel_number"] = relations["rel_name"].str[1:].astype(int)
    relations = combine_relations_with_multiple_triggers(relations, entities, config)

    # Add degree info to relations
    relations["degree"] = pd.Series(relations.index).map(lambda rel: degrees.query("rel_name == @rel")["degree"].to_list()).str[0].to_list()
    relations["degree"].fillna("Facilitate", inplace=True) # for missing relations

    # add relation info to entities
    used_causal_arguments = [arg for arg in config["causal_arguments"] if arg in relations.columns]
    for arg in used_causal_arguments:
        relations_args = relations[[arg, "rel_name"]].explode(arg).groupby(arg).agg(list).reset_index()
        relations_args = relations_args.rename({"rel_name": f"{arg}_rels", arg: "ent_name"}, axis=1)
        entities = pd.merge(entities, relations_args, on="ent_name", how="left")

    # align relations with text
    with open(txt_path, "r") as f:
        text = nlp(f.read())

    tokens = [token.text for token in text]
    token_pos = np.array([token.idx for token in text])

    if sentence_split_pattern == "": 
        # no split pattern -> use spacy sentences
        splits = [sent.end for sent in text.sents]
    else: 
        # use split pattern
        splits = np.where([sentence_split_pattern in token for token in tokens])[0]
    
    # add start and end
    splits = np.concatenate([[0], splits, [len(tokens)+1]]) 

    # fix splits that occur within relation
    relations["boundaries"] = relations[used_causal_arguments].apply(lambda args: get_relation_boundaries(args.values, entities, token_pos), axis=1)
    within_relation_array = np.array([False] * (max(splits)+1))
    
    for b_min, b_max in relations["boundaries"]:
        within_relation_array[b_min:b_max] = True
    
    splits = np.array([split for split in splits if not within_relation_array[split]])

    # split text into smaller chunks
    entities["tokens_pos"] = get_token_positions_for_entities(entities, token_pos)
    entities["sentence"] = -1
    entities["tokens_pos_sentence"] = -1
    sentence_tokens = []
    for i in range(len(splits)-1):

        if i > 0:
            split_start = splits[i] + len(sentence_split_pattern)
        else: # no splitting at first token
            split_start = splits[i]

        split_end = splits[i+1]
        start_pos = entities["tokens_pos"].str[0]

        sentence_entities = (split_start <= start_pos) & (start_pos < split_end)
        entities.loc[sentence_entities, "sentence"] = i
        entities.loc[sentence_entities, "tokens_pos_sentence"] = entities["tokens_pos"].map(lambda l: [pos - split_start for pos in l])

        sentence_tokens.append(tokens[split_start:split_end])

    return {
        "entities": entities,
        "relations": relations,
        "degrees": degrees,
        "coreferences": coreferences,
        "sentence_tokens": sentence_tokens,
    }


def get_token_positions_for_entities(entities, token_pos):
    """Get absolute positions of tokens of all entities in text

    Parameters
    ----------
    entities : pd.DataFrame
        Contains entitiy information
    token_pos : np.array
        Array with all token start positions

    Returns
    -------
    list
        List containing the absolute token positions of the entities in text
    """
    tokens_pos = []
    for spans in tqdm(entities["spans"].to_list()):
        tokens_pos_ent = []
        for ent_start, ent_end in spans:
            token_start = np.sum(token_pos < int(ent_start))
            token_end = np.sum(token_pos < int(ent_end)-1)

            if token_start == token_end: # edge case
                token_start -= 1

            tokens_pos_ent += list(range(token_start, token_end))

        tokens_pos.append(tokens_pos_ent)

    return tokens_pos


def get_relation_boundaries(rel_entities, entities, token_pos):
    """Compute the spans for each relation.
    Given by start of first entitiy and end of last entity.
    Used to fix spans where relations are disconnected.

    Parameters
    ----------
    rel_entities : list
        All entities within a relation
    entities : pd.DataFrame
        Contains entity information
    token_pos : np.array
        Array with all token start positions

    Returns
    -------
    (int, int)
        Tuple of start and end of relation
    """
    spans = []
    for ent in flatten(rel_entities):
        ent_boundaries = flatten(entities.query("ent_name == @ent")["spans"].iloc[0])
        spans += [np.sum(token_pos < int(ent_boundary)) for ent_boundary in ent_boundaries]

    return min(spans), max(spans)


def combine_relations_with_multiple_triggers(relations, entities, config):
    """Combine arguments for relations with the same Trigger groups.
    Example sentence in Fondsforste: "Während der beherzte Weidmann..."

    Parameters
    ----------
    relations : pd.DataFrame
        Contains relation information
    entities : pd.DataFrame
        Contains entitiy information
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    pd.DataFrame
        Relation data where relations with same Trigger are combined
    """

    # find trigger/relation for all triggers with same span
    relations_tmp = pd.DataFrame(relations["Trigger"].str[0])
    relations_tmp["Relation"] = relations_tmp.index
    relations_tmp = pd.merge(relations_tmp, entities[["spans", "ent_name"]].rename({"ent_name": "Trigger"}, axis=1), how="left", on="Trigger")
    relations_tmp["spans"] = relations_tmp["spans"].map(lambda l: "|".join([str(i) for i in flatten(l)]))
    rel_for_trigger = relations_tmp.groupby("spans").agg(lambda g: list(set(g)))

    # make new relations for combined triggers
    new_relations = []
    for rels_to_combine in rel_for_trigger["Relation"].to_list():
        rels = relations.loc[rels_to_combine]

        # fill nans with empty lists for combine step
        rels = rels.applymap(lambda item: item if item == item else [])

        new_relation = rels.iloc[0].copy(deep=True)

        for arg in config["causal_arguments"]:
            if arg not in rels.columns: continue
            new_relation[arg] = sorted(set(flatten(rels[arg].fillna(list))))

        new_relations.append(new_relation)

    return pd.DataFrame(new_relations).sort_values("rel_number")


def get_sentence_data(relations, sentence_tokens, entities, coreferences, config):
    """Create sentence data where each sample includes all necessary 
    information. 
    - Incorporates Coreferences
    - Collects positions of causal arguments, type and degree
    - Removes overlapp of causal arguments with Triggers
    - Removes newlines in tokens

    Parameters
    ----------
    relations : pd.DataFrame
        Information about relations
    sentence_tokens : list
        Tokens of text split into sentences
    entities : pd.DataFrame
        Information about entities
    coreferences : pd.DataFrame
        Information about coreferences
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List of sentence samples with tokens and relations
    """
    
    coreferences_reverse = coreferences.copy()
    coreferences_reverse[["ref1", "ref2"]] = coreferences_reverse[["ref2", "ref1"]]

    coreference_lookup = pd.concat([coreferences, coreferences_reverse]).set_index("ref1")["ref2"].to_dict()

    sentence_data_dict = defaultdict(dict)

    for i, row in relations.iterrows():
        
        relation_data = defaultdict(list)
        for arg in config["causal_arguments"]:
            if arg not in relations.columns or len(row[arg]) == 0: continue
            
            for ent in row[arg]:
                sentence_num, pos_tokens = entities.loc[entities["ent_name"] == ent, ["sentence", "tokens_pos_sentence"]].values[0]

                if arg == "Trigger":
                    relation_data[arg] += sorted(pos_tokens)
                else:
                    relation_data[arg].append(sorted(pos_tokens))

                # check for coreferences (only use anaphoras for now)
                if ent in coreference_lookup:
                    referenced_ent = coreference_lookup[ent]
                    sentence_num_reference, pos_tokens_reference = entities.loc[entities["ent_name"] == referenced_ent, ["sentence", "tokens_pos_sentence"]].values[0]
                    
                    if sentence_num == sentence_num_reference and max(pos_tokens_reference) < min(pos_tokens):
                        relation_data["coreference"].append((pos_tokens_reference, pos_tokens))

        relation_data["type"] = row["rel_type"][0]
        sentence_data_dict[sentence_num][row["rel_name"]] = dict(relation_data)

    # bring data into final form
    sentence_data = []
    for i in range(len(sentence_tokens)):
        tokens = sentence_tokens[i]

        if i in sentence_data_dict:
            sent_relations = sentence_data_dict[i]
            for rel in sent_relations:

                degree = relations.loc[rel]["degree"]
                sent_relations[rel]["degree"] = degree
                
            sentence_data.append({
                "tokens": tokens,
                "relations": list(sent_relations.values()),
                "is_causal": True,
            })
        else:
            sentence_data.append({
                "tokens": tokens,
                "relations": [{'type': 'None', 'degree': 'None'}],
                "is_causal": False,
            })

    # remove overlap of arguments with trigger
    for sample in sentence_data:
        for rel in sample["relations"]:
            for arg in config["causal_arguments"]:
                if arg == "Trigger" or arg not in rel: continue

                rel[arg] = [sorted(set(rel_arg_ent) - set(rel["Trigger"])) for rel_arg_ent in rel[arg] if len(set(rel_arg_ent) - set(rel["Trigger"])) > 0]

                # if arg has no more tokens, drop arg
                if len(rel[arg]) == 0:
                    del rel[arg]

    # remove new_line tokens 
    # (make problems with alignment of labels with tokenizer)
    # align labels to account for missing newlines
    for sample in sentence_data:

        new_tokens = []
        new_line_positions = []
        for i, token in enumerate(sample["tokens"]):
            if re.match("^[\n\t\s]+$", token) != None:
                new_line_positions.append(i)
            else:
                new_tokens.append(token)
        sample["tokens"] = new_tokens

        for rel in sample["relations"]:
            for arg in config["causal_arguments"]:
                if arg not in rel: continue

                if arg == "Trigger":
                    rel[arg] = remove_newline_from_ent(new_line_positions, rel[arg])
                else:
                    for i in range(len(rel[arg])):
                        rel[arg][i] = remove_newline_from_ent(new_line_positions, rel[arg][i])

    return sentence_data


def remove_newline_from_ent(newlines, positions):
    """Remove newlines and fix resulting label mismatch,

    Parameters
    ----------
    newlines : list
        Positions of newlines in sample
    positions : list
        Positions of entities in sample

    Returns
    -------
    list
        List of positions of entities with removed newlines
    """
    positions = np.array(positions)

    for new_line_pos in reversed(newlines):
        positions[positions>=new_line_pos] = positions[positions>=new_line_pos] - 1

    return positions.tolist()


def get_labels_for_sentence(tokens, relations, add_coreferences=False):
    """Converts tokens with relation data into BIO labels.
    If more than 1 relation, returned labels contain multiple lists of labels.

    Parameters
    ----------
    tokens : list
        List of tokens in the sample
    relations : list
        Relations in the sample
    add_coreferences : bool, optional
        Whether coreferenced arguments should be labeled, by default False

    Returns
    -------
    list
        Labels for each relation in the sample in BIO format
    """

    all_labels = []
    for relation in relations:
        labels = np.array(["O"] * len(tokens), dtype='<U20')
        for label, positions in relation.items():

            # not a label
            if label == "type" or label == "degree" or label == "coreference": 
                continue

            # For Trigger we always use "B" because they are combined in Task 2
            if label == "Trigger":
                labels[positions] = "B-Trigger"
                continue

            # other causal arguments
            for position in positions:
                if add_coreferences and "coreference" in relation:
                    # coreferences are sorted by position (we only use anaphora)
                    for ref1_positions, ref2_positions in relation["coreference"]:
                        if len(set(position) - set(ref2_positions)) == 0:
                            position = sorted(position + ref1_positions)

                position = np.array(position)
                begin_tags = [True] + list(position[:-1] + 1 != position[1:])

                position_labels = np.array([f"I-{label}"] * len(position), dtype='<U20')
                position_labels[begin_tags] = f"B-{label}"

                labels[position] = position_labels

        all_labels.append(labels)

    return all_labels


def sanity_check(idx, sentence_data, add_coreferences):
    """Retrives tokens and all NER labels for a sentence.

    Parameters
    ----------
    idx : int
        Id of sentence
    sentence_data : list
        List of sentence samples with tokens and relations
    add_coreferences : bool
        Whether coreferences should be added

    Returns
    -------
    pd.DataFrame
        Tokens and Labels in DataFrame format for easier inspection
    """
    tokens = sentence_data[idx]["tokens"]
    relations = sentence_data[idx]["relations"]
    labels = get_labels_for_sentence(tokens, relations, add_coreferences=add_coreferences)

    sanity_df = pd.DataFrame({
        "tokens": tokens,
    })

    for i, label_set in enumerate(labels):
        sanity_df[f"labels{i}"] = label_set

    return sanity_df


def postprocessing_predictions(args_result, config, sim_threshold=0.5, remove_duplicates=False):
    """Basic postprocessing of predictions.
    - Removes relations without any causal arguments other than Trigger
    - Optionally, removes relations with a high overlap in tokens

    Parameters
    ----------
    args_result : pd.DataFrame
        Contains prediction results
    config : dict
        Dictionary containing overall parameters and constants
    sim_threshold : float, optional
        Threshold for dropping relations with high overlap, by default 0.5
    remove_duplicates : bool, optional
        Whether relations with high overalp should be removed, by default False

    Returns
    -------
    pd.DataFrame
        Prediction results after postprocessing
    """

    # remove relations without causal arguments
    args_result = args_result[args_result["labels"].apply(lambda labels: not all([l in ["O", "B-Trigger"] for l in labels]) ).values]

    # remove suspected duplicates (experimental)
    if remove_duplicates:
        relations_to_remove = []
        for sent_id, sample_data in args_result.groupby("id"):
            
            for i, j in itertools.combinations(np.arange(len(sample_data)), 2):
                ent1 = np.array(remove_prefix(sample_data.iloc[i]["labels"]))
                ent2 = np.array(remove_prefix(sample_data.iloc[j]["labels"]))

                # only focus on causal args other than Trigger
                ent1[ent1 == "Trigger"] = "O"
                ent2[ent2 == "Trigger"] = "O"
                prefix_no_other = list(set(remove_prefix(config["label_list"])) - set("O") | set(["Trigger"]))

                f1 = f1_score(ent1, ent2, labels=prefix_no_other, average="micro", zero_division=False)

                if f1 >= sim_threshold:
                    
                    # remove relation with less arguments
                    if sum(ent1 != "O") > sum(ent2 != "O"):
                        relation_id = f"{sent_id}_{j}"
                    else:
                        relation_id = f"{sent_id}_{i}"

                    relations_to_remove.append(relation_id)
        
        valid_relations = args_result.apply(lambda r: f"{r['id']}_{r['relation_id']}" not in relations_to_remove, axis=1).values
        args_result = args_result[valid_relations]

    return args_result


def add_CAB_tokens(spellnorm_data, sentence_data):
    """Align text tokens to normalized CAB tokens
    Many edge cases due to changed splitting of tokens and whitespace treatment
    between spacy and CAB
    CAB: https://www.deutschestextarchiv.de/demo/cab/file

    Parameters
    ----------
    spellnorm_data : list
        List of json objects containing the normalized tokens
    sentence_data : list
        List of sentence samples with tokens and relations

    Returns
    -------
    list
        sentence_data with additional normalized tokens
    """

    spell_orig_tokens = flatten([list(map(lambda d: d["text"], spellnorm_data[i]["tokens"])) for i in range(len(spellnorm_data))])
    spell_corr_tokens = flatten([list(map(lambda d: d["moot"]["word"], spellnorm_data[i]["tokens"])) for i in range(len(spellnorm_data))])

    spell_token_ctr = 0
    for sent_data in sentence_data:

        normalized_tokens_sent = []
        for token in sent_data["tokens"]:

            # edge cases with whitespace in front
            num_whitespace = 0
            for char in token:
                if char == " " or char == "\n": num_whitespace += 1
                else: break

            if num_whitespace == len(token):
                normalized_tokens_sent.append(token)
                continue

            # edge case with tokens consisting of "-"
            if (token == "--" and spell_orig_tokens[spell_token_ctr][:2] != "--") or \
            (token == "---" and spell_orig_tokens[spell_token_ctr][:3] != "---"): 
                normalized_tokens_sent.append(token)
                continue

            # get token from spellnorm
            spell_orig_token = spell_orig_tokens[spell_token_ctr]
            spell_corr_token = spell_corr_tokens[spell_token_ctr]

            # boundaries are different, iterate over several spellnorm tokens to 
            # find full match
            inc_counter = True
            while spell_orig_token != token:
                if len(spell_orig_token) > len(token):

                    if token.endswith("-") and spell_orig_token[len(token)-1] != "-": 
                        token = token[:-1]

                    spell_orig_token = spell_orig_token[:len(token)]
                    spell_corr_token = spell_corr_token[:len(token)]

                    # remove partial match from spellnorm tokens
                    spell_orig_tokens[spell_token_ctr] = spell_orig_tokens[spell_token_ctr][len(token):]
                    spell_corr_tokens[spell_token_ctr] = spell_corr_tokens[spell_token_ctr][len(token):]

                    inc_counter = False
                    break

                else:
                    assert spell_orig_token == token[:len(spell_orig_token)]

                    spell_token_ctr += 1
                    new_spell_orig_token = spell_orig_tokens[spell_token_ctr]
                    new_spell_corr_token = spell_corr_tokens[spell_token_ctr]

                    new_spell_pos = 0
                    for token_pos in range(len(spell_orig_token), len(token)):
                        
                        # print(new_spell_pos, token_pos)
                        if token[token_pos] == " " or token[token_pos] == "\n":
                            spell_orig_token += token[token_pos]
                            spell_corr_token += token[token_pos]

                        if token[token_pos] == "-" and new_spell_orig_token[new_spell_pos] != "-":
                            spell_orig_token += "-"

                        else:
                            if new_spell_pos < len(new_spell_orig_token):

                                assert new_spell_orig_token[new_spell_pos] == token[token_pos]
                                spell_orig_token += new_spell_orig_token[new_spell_pos]

                                # orig and corr can be different length
                                if new_spell_pos < len(new_spell_corr_token):
                                    spell_corr_token += new_spell_corr_token[new_spell_pos]
                                
                                new_spell_pos += 1
                            else:
                                spell_corr_token += new_spell_corr_token[new_spell_pos:] # fully matched original, add rest of spell_corr_token
                                break

                    if new_spell_pos < len(new_spell_orig_token):
                        inc_counter = False
                        spell_orig_tokens[spell_token_ctr] = spell_orig_tokens[spell_token_ctr][new_spell_pos:]
                        spell_corr_tokens[spell_token_ctr] = spell_corr_tokens[spell_token_ctr][new_spell_pos:]

            assert token == spell_orig_token

            # only change words and not characters
            if not re.search("[a-zA-Z]", spell_corr_token):
                spell_corr_token = spell_orig_token

            if inc_counter: spell_token_ctr += 1
            normalized_tokens_sent.append(spell_corr_token)

        sent_data["normalized_tokens"] = normalized_tokens_sent

    return sentence_data


def get_Rehbein_data(data_path, config):
    """Load and convert Rehbein data from preprocessed json file.
    Additionally add degrees, types and fill in missing values.

    Parameters
    ----------
    data_path : str
        Location of the preprocessed json file with the data of Rehbein
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List of sentence samples with tokens and relations of Rehbein
    """

    # load pre-processed rehbein json file
    raw_data = pd.read_json(data_path + "data/data_causal_arguments.json")
    raw_data = raw_data[~(raw_data["causal"] & raw_data["annotations_args_tokenized"].map(lambda l: sum(np.array(l)=="B-Trigger") == 0))] # remove sentences without trigger (probably erronous annotation)
    raw_data["annotations_args_tokenized"] = raw_data["annotations_args_tokenized"].map(lambda l: [i if i[2:] != "Controller" else i[:2]+"Controlling" for i in l]) # change "Controller" to "Controlling"
    raw_data = raw_data.reset_index(drop=True)

    degree_info = {}
    for pos_type in ["adpositions", "nouns", "verbs"]:
        pos_type_df  = read_ods(data_path + "lexicon/CC_lexicon.ods", pos_type)[["Lemma", "Degree"]]
        pos_type_df["Lemma"] = pos_type_df["Lemma"].str.lower()
        pos_type_dict = pos_type_df.set_index("Lemma").to_dict()["Degree"]
        degree_info[pos_type[:-1]] = pos_type_dict # remove "s" because of differences to raw_df

    # manually added (not captured by lemmatization), luckily all are "facilitate"
    for adpos in ["zweck", "danken", "weg", "für", "trotzen", "halb", "über", "vom"]: # makes "Wegen" at beginning of sentence to "weg"
        degree_info["adposition"][adpos] = "facilitate"

    for noun in ["motiv", "stimuli", "auslöser", "anlaß", "spätfolge", "hauptursachen"]:
        degree_info["noun"][noun] = "facilitate"

    for verb in ["hängen", "zielen", "packen", "richten", "bringen", "tauchen", "treten", "tritt", "tun", "werfen", "wirken", "tragen", "bereit", "stellen", "fahren", "geben", "beschwören", "rufen", "laufen", "nehmen", "kommen", "lässt", "leiten", "liegen", "schlagen", "stecken", "getrieben", "stehen", "veranlasst", "veranlassten", "veranlasste", "ziehen", "fügen", "haben"]:
        degree_info["verb"][verb] = "facilitate"

    @functools.lru_cache(500) # nlp call takes forever...
    def lemmatize(word):
        return config["nlp_ger"](str(word))[0].lemma_.lower() 

    raw_data["Trigger_lemma"] = raw_data.apply(lambda row: lemmatize(np.array(row["sentences_tokenized"])[np.array(row["annotations_type_tokenized"]) != "O"][0]) if row["causal"] else "-", axis=1) # extracts lemma of trigger
    raw_data["pos_type"] = raw_data["pos_type"].replace("prep", "adposition")
    raw_data["degree"] = raw_data.apply(lambda row: degree_info[row["pos_type"]][row["Trigger_lemma"]] if row["Trigger_lemma"] in degree_info[row["pos_type"]] else "-", axis=1)

    assert len(raw_data.query("degree == '-'")[["pos_type", "Trigger_lemma"]].drop_duplicates()) == 3 # "-" for noun, adpos and verb

    # bring into correct form
    rehbein_sentence_tokens = raw_data["sentences_tokenized"].to_list()
    rehbein_sentence_data = []

    # parse labeled data to sentence_data format
    for i, (annotated_args, causal_type, degree, is_causal) in raw_data[["annotations_args_tokenized", "causal_type", "degree", "causal"]].iterrows():

        if is_causal:
            relation_data = {}
            relation_data["type"] = causal_type
            relation_data["degree"] = degree.title()
        else:
            relation_data = {
                "type": "None",
                "degree": "None",
            }

        annotated_args_no_prefix = set([t if t=="O" else t[2:] for t in annotated_args]) - {"O"}
        for arg in annotated_args_no_prefix: relation_data[arg] = []

        l_with_b = ""
        l_with_b_positions = []
        for l_idx, l in enumerate(annotated_args + ["O"]): # + ["O"] to also save last entity
            if l[0] == "B":
                if len(l_with_b_positions): relation_data[l_with_b].append(l_with_b_positions)
                l_with_b = l[2:]
                l_with_b_positions = [l_idx]

            elif l[0] == "I" and l[2:] == l_with_b:
                l_with_b_positions.append(l_idx)

            else:
                if len(l_with_b_positions): relation_data[l_with_b].append(l_with_b_positions)
                l_with_b = ""
                l_with_b_positions = []

        if "Trigger" in relation_data:
            relation_data["Trigger"] = flatten(relation_data["Trigger"])

        if "None" in relation_data: del relation_data["None"]

        rehbein_sentence_data.append({
            "is_causal": is_causal,
            "relations": [relation_data],
            "tokens": rehbein_sentence_tokens[i],
        })

    return rehbein_sentence_data


def get_dunietz_data(data_path, config):
    """Load and convert Dunietz data
    Ignores other relations than causal (temporal, correlation...)

    Parameters
    ----------
    data_path : str
        Location of Folder containing Duniez data
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List of sentence samples with tokens and relations of Dunietz
    """
    dunietz_sentence_data = []

    # not all data available
    for subcorpus in ["/MASC", "/CongressionalHearings"]: 
        files = sorted(set([file_name.split(".")[0] for file_name in os.listdir(data_path+subcorpus) if file_name.endswith(".ann")]))

        for file in tqdm(files):
            file_path = data_path + subcorpus + "/" + file
            dunietz_brat_data = get_brat_data(config, ann_path=file_path+".ann", txt_path=file_path+".txt", nlp=config["nlp_eng"], sentence_split_pattern="")
            dunietz_sentence_data_file = get_sentence_data(dunietz_brat_data["relations"], dunietz_brat_data["sentence_tokens"], dunietz_brat_data["entities"], dunietz_brat_data["coreferences"], config)
            
            dunietz_sentence_data += dunietz_sentence_data_file

    return dunietz_sentence_data


def get_fondsforste_data(data_path, config):
    """Load and convert Fondsforste data
    Ignores other relations than causal (temporal, correlation...)

    Parameters
    ----------
    data_path : str
        Location of Folder containing Fondsforste data
    config : dict
        Dictionary containing overall parameters and constants


    Returns
    -------
    list
        List of sentence samples with tokens and relations of Fondsforste
    """

    fondsforste_txt_path = data_path + "Staats-_und_Fondsforste_Ausstellung_Paris_1900_Leitfaden_2_0.txt"
    fondsforste_ann_path = data_path + "Staats-_und_Fondsforste_Ausstellung_Paris_1900_Leitfaden_2_0.ann"

    fondsforste_json_path_norm = data_path + "Staats-_und_Fondsforste_Ausstellung_Paris_1900_Leitfaden_2_0_spellnorm.json"

    with open(fondsforste_json_path_norm, "r") as f:
        spellnorm_data = json.load(f)["body"]

    brat_data = get_brat_data(config, 
                                ann_path=fondsforste_ann_path, 
                                txt_path=fondsforste_txt_path, 
                                nlp=config["nlp_ger"])

    fondsforste_sentence_data = get_sentence_data(
        brat_data["relations"], 
        brat_data["sentence_tokens"], 
        brat_data["entities"], 
        brat_data["coreferences"], 
        config
    )

    fondsforste_sentence_data = add_CAB_tokens(spellnorm_data, fondsforste_sentence_data)

    return fondsforste_sentence_data


def get_forstvermessung_data(data_path, config):
    """Load and convert Forstvermessung data
    Ignores other relations than causal (temporal, correlation...)

    Parameters
    ----------
    data_path : str
        Location of Folder containing Forstvermessung data
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List of sentence samples with tokens and relations of Forstvermessung
    """
    spellnorm_data = []
    forstvermessung_sentence_data = []

    for fv_part in range(1, 4):
        forstvermessung_txt_path = data_path + f"Instruction_Forstvermessung_Wien_1878_{fv_part}.txt"
        forstvermessung_ann_path = data_path + f"Instruction_Forstvermessung_Wien_1878_{fv_part}.ann"
        forstvermessung_json_path_norm = data_path + f"Instruction_Forstvermessung_Wien_1878_{fv_part}_spellnorm.json"
        
        with open(forstvermessung_json_path_norm, "r") as f:
            spellnorm_data += json.load(f)["body"]

        brat_data = get_brat_data(config, ann_path=forstvermessung_ann_path, txt_path=forstvermessung_txt_path, nlp=config["nlp_ger"], sentence_split_pattern="")

        forstvermessung_sentence_data += get_sentence_data(
            brat_data["relations"], 
            brat_data["sentence_tokens"], 
            brat_data["entities"], 
            brat_data["coreferences"], 
            config
        )

    forstvermessung_sentence_data = add_CAB_tokens(spellnorm_data, forstvermessung_sentence_data)
    return forstvermessung_sentence_data


def fondsforste_sanity_check(fondsforste_sentence_data):
    """Sanity check of selected samples for Fondsforste
    To check input data, loading and pre-processing steps

    Parameters
    ----------
    data_path : list
        List of sentence samples with tokens and relations of Fondsforste

    Returns
    -------
    """
    try:
        df = sanity_check(7, fondsforste_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'O', 'O', 'O', 'B-Effect', 'B-Trigger', 'O']

        df = sanity_check(10, fondsforste_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Affected', 'B-Controlling', 'I-Controlling', 'I-Controlling', 'B-Cause', 'I-Cause', 'I-Cause', 'B-Effect', 'I-Effect', 'I-Effect', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'O']

        df = sanity_check(12, fondsforste_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'B-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'O', 'O', 'B-Affected', 'B-Effect', 'I-Effect', 'I-Effect', 'O']
        assert df["labels3"].to_list() == ['O', 'O', 'B-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Affected', 'B-Effect', 'I-Effect', 'I-Effect', 'O']

        df = sanity_check(77, fondsforste_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Actor', 'I-Actor', 'O', 'B-Cause', 'B-Trigger', 'B-Trigger', 'O', 'O', 'O', 'O', 'O', 'O']
        assert df["labels1"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Cause', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O']

        df = sanity_check(1, fondsforste_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O']

        df = sanity_check(8, fondsforste_sentence_data, True)
        assert df["labels0"].to_list() == ['O', 'O', 'B-Affected', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Affected', 'I-Affected', 'I-Affected', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'B-Trigger', 'B-Trigger', 'B-Trigger', 'O']
    except Exception as e:
        print("Something wrong with SF data...")
        traceback.print_exc()
        return

    print("SF data correct")


def forstvermessung_sanity_check(forstvermessung_sentence_data):
    """Sanity check of selected samples for Forstvermeesung
    To check input data, loading and pre-processing steps

    Parameters
    ----------
    data_path : list
        List of sentence samples with tokens and relations of Forstvermeesung

    Returns
    -------
    """
    try:

        df = sanity_check(302, forstvermessung_sentence_data, False)
        assert df["labels0"].to_list() == ['B-Controlling', 'I-Controlling', 'I-Controlling', 'I-Controlling', 'I-Controlling', 'I-Controlling', 'I-Controlling', 'O', 'B-Trigger', 'B-Actor', 'I-Actor', 'I-Actor', 'I-Actor', 'I-Actor', 'I-Actor', 'I-Actor', 'O', 'B-Trigger', 'B-Trigger', 'B-Trigger', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O']

        df = sanity_check(303, forstvermessung_sentence_data, False)
        assert df["labels0"].to_list() == ['B-Trigger', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        assert df["labels1"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O']

        df = sanity_check(402, forstvermessung_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'I-Cause', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'B-Trigger', 'O']

        df = sanity_check(1110, forstvermessung_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O', 'B-Trigger', 'B-Controlling', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O'] 

    except Exception as e:
        print("Something wrong with FV data...")
        traceback.print_exc()
        return

    print("FV data correct")


def rehbein_sanity_check(rehbein_sentence_data):
    """Sanity check of selected samples for Rehbein
    To check input data, loading and pre-processing steps

    Parameters
    ----------
    data_path : list
        List of sentence samples with tokens and relations of Rehbein

    Returns
    -------
    """
    try:
        df = sanity_check(7, rehbein_sentence_data, False)
        assert df["labels0"].to_list() == ['B-Trigger', 'B-Cause', 'I-Cause', 'I-Cause', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

        df = sanity_check(10, rehbein_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'B-Trigger', 'B-Cause', 'I-Cause', 'O', 'O', 'O']

    except Exception as e:
        print("Something wrong with Rehbein data...")
        traceback.print_exc()
        return

    print("Rehbein data correct")


def dunietz_sanity_check(dunietz_sentence_data):
    """Sanity check of selected samples for Dunietz
    To check input data, loading and pre-processing steps

    Parameters
    ----------
    data_path : list
        List of sentence samples with tokens and relations of Dunietz

    Returns
    -------
    """
    try:
        df = sanity_check(3, dunietz_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O']
        assert df["labels1"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Cause', 'O', 'O', 'O', 'O', 'B-Trigger', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O']
        assert df["labels2"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'B-Trigger', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O']


        df = sanity_check(1522, dunietz_sentence_data, False)
        assert df["labels0"].to_list() == ['O', 'O', 'O', 'O', 'O', 'O', 'B-Trigger', 'O', 'B-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'I-Effect', 'O', 'B-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'I-Cause', 'O']
    except Exception as e:
        print("Something wrong with Dunietz data...")
        traceback.print_exc()
        return

    print("Dunietz data correct")


def get_test_predictions(predictions, dataset, config):
    """Aggregates results of all tasks from predictions.

    Parameters
    ----------
    predictions : dict
        Dict of lists containing predictions for each task
    dataset : torch.Dataset
        Dataset containing the input data
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Tuple containing Trigger results (Tasks 2&3) and Argument results
        (Tasks 4&5)
    """
   
    arg_results = []
    trigger_results = []
    for sample_num, sample in enumerate(dataset):

        sample_id = sample["sent_id"]
        tokens = dataset.__gettokens__(sample_id)
        num_relations = len(dataset.sentence_data[sample_id]["relations"])

        # 0 is CLS, -1 is SEP
        range_slice = slice(1, np.sum(sample["attention_masks"]) - 1)
        input_ids = np.array(sample["input_ids"][range_slice])
        
        # to account for split words
        word_ids = []
        for i, token in enumerate(tokens):
            word_ids += (len(dataset.tokenizer.encode(token)) - 2) * [i]  

        # ---- Trigger results ----
        true_triggers = sample['sample_triggers']
        true_triggers = [config["label_list"][l] for l in true_triggers[range_slice]]
        pred_triggers = predictions['pred_triggers'][sample_num]
        pred_triggers = [config["label_list"][l] for l in pred_triggers[range_slice]]

        true_trigger_decoded = get_ents_for_sample(true_triggers, word_ids)
        true_trigger_labels = get_labels_for_ents(tokens, true_trigger_decoded)

        pred_trigger_decoded = get_ents_for_sample(pred_triggers, word_ids)
        pred_trigger_labels = get_labels_for_ents(tokens, pred_trigger_decoded)

        # ---- trigger combinations ----
        true_trigger_groups = make_trigger_groups(
            torch.tensor(sample['sample_relation_arguments'][:num_relations]), 
            torch.tensor(sample["sample_subwords"]),
            config
        )
        _, true_combine_triggers = get_trigger_comb_input(true_trigger_groups)
        true_combine_triggers = np.array(true_combine_triggers, dtype=bool)

        pred_combine_triggers = predictions['pred_combine_triggers'][sample_num]

        trigger_result = pd.DataFrame({
            "tokens": [tokens],
            "true_trigger": [list(true_trigger_labels)],
            "pred_trigger": [list(pred_trigger_labels)],
            "true_combine_triggers": [true_combine_triggers],
            "pred_combine_triggers": [pred_combine_triggers],
            "true_is_causal": sum(true_trigger_labels == "B-Trigger") > 0,
            "pred_is_causal": sum(pred_trigger_labels == "B-Trigger") > 0,
            "id": [sample_id],
        })
        trigger_results.append(trigger_result)

        # ---- arguments results ----
        true_args = sample['sample_relation_arguments'][:num_relations]
        pred_args = predictions['pred_args'][sample_num]

        # ---- type/degree results ----
        true_type = sample['sample_relation_types'][:num_relations]
        pred_type = predictions['pred_type'][sample_num]

        true_degree = sample['sample_relation_degrees'][:num_relations]
        pred_degree = predictions['pred_degree'][sample_num]

        # bring results in correct form
        arg_results += get_argument_sample_result(tokens, pred_args, pred_type, pred_degree, word_ids, "Predicted", sample_id, range_slice, config)
        arg_results += get_argument_sample_result(tokens, true_args, true_type, true_degree, word_ids, "Ground Truth", sample_id, range_slice, config)

        # add dummy results if no argument is in GT or predicted
        # Predicted non-causal
        if len(pred_args) == 0:
            arg_results.append(get_dummy_result(tokens, "Predicted", sample_id))
    
        # GT non-causal
        if len(true_args) == 0:
            arg_results.append(get_dummy_result(tokens, "Ground Truth", sample_id))

    arg_results = pd.concat(arg_results).reset_index(drop=True)
    trigger_results = pd.concat(trigger_results).reset_index(drop=True)
    return arg_results, trigger_results


def get_ents_for_sample(prediction, word_ids):
    """Gets entities from BERT predictions (needs to account for subwords)

    Parameters
    ----------
    prediction : list
        Label predictions for each input_id
    word_ids : list
        Mapping of labels to word in input to account for subwords

    Returns
    -------
    pd.DataFrame
        Prediction results with entities and their positions
    """

    preds = []
    current_pred = []
    label_with_B = ""

    for label, word_id in zip(prediction, word_ids):
        if label == "O":
            if current_pred:
                preds.append((label_with_B, current_pred))
            current_pred = []
            label_with_B = ""
            continue

        current_label_type = label[2:]

        # if new "B" is found, store label
        if label[0] == "B":
            # filter out duplicate starts due to split tokens
            if current_pred and word_id == current_pred[-1]: continue 

            if current_pred:
                preds.append((label_with_B, current_pred))

            current_pred = [word_id]
            label_with_B = current_label_type

        # append only if I and same label
        if label[0] == 'I':
            if label_with_B == current_label_type:
                current_pred.append(word_id)
            else:
                if current_pred:
                    preds.append((label_with_B, current_pred))
                current_pred = []
                label_with_B = ""

    # append leftover
    if len(current_pred) and label_with_B != "":
        preds.append((label_with_B, current_pred))

    pred_entities = pd.DataFrame(preds, columns=["class", "token_pos"])

    # remove duplicates that occur when for a single word multiple "B-" tags are
    # predicted with a different "I-" tag in between
    pred_entities["token_pos"] = pred_entities["token_pos"].apply(lambda l: sorted(np.unique(l)))
    pred_entities["token_pos_start"] = pred_entities["token_pos"].str[0]
    pred_entities = pred_entities.drop_duplicates("token_pos_start").drop("token_pos_start", axis=1)

    return pred_entities


def get_dummy_result(tokens, kind, sample_id):
    """Placeholder Results if nothing is GT or predicted

    Parameters
    ----------
    tokens : list
        List of tokens
    kind : str
        Used for denoting GT or prediction
    sample_id : int
        Id of sample

    Returns
    -------
    pd.DataFrame
        Dummy result
    """
    return pd.DataFrame({
        "tokens": [tokens],
        "labels": [["O"] * len(tokens)],
        "kind": [kind],
        "type": ["None"],
        "degree": ["None"],
        "relation_id": [0],
        "id": [sample_id],
    })


def get_labels_for_ents(tokens, entities):
    """Construct label list from position of entities.

    Parameters
    ----------
    tokens : list
        List of tokens in sample
    entities : pd.DataFrame
        Entity information about entities in sample

    Returns
    -------
    list
        BIO labels for entities in sample
    """
    labels = np.array(["O"] * len(tokens), dtype='<U20')
    for _, (ent_type, token_pos) in entities.iterrows():
        labels[token_pos[0]] = f"B-{ent_type}"
        labels[token_pos[1:]] = f"I-{ent_type}"
    
    return labels


def get_argument_sample_result(tokens, args, types, degrees, word_ids, kind, sample_id, range_slice, config):
    """Align argument labels with type, degree and meta information for each
    relation in sample

    Parameters
    ----------
    tokens : list
        List of tokens in sample
    args : list
        Arguments for all relations in sample
    types : list
        Types for all relations in sample
    degrees : list
        Degrees for all relations in sample
    word_ids : list
        Mapping of labels to word in input to account for subwords
    kind : str
        Used for denoting GT or prediction
    sample_id : int
        Id of sample
    range_slice : slice
        Slice object denoting the relevant positions of input_ids (no CLS or SEP)
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        List containing the relation results for sample
    """
    sample_res = []
    for rel_i in range(len(args)):

        rel_args = args[rel_i]
        rel_args = [config["label_list"][l] for l in rel_args[range_slice]]

        rel_args_decoded = get_ents_for_sample(rel_args, word_ids)
        labels = get_labels_for_ents(tokens, rel_args_decoded)

        rel_result = {
            "tokens": [tokens],
            "labels": [labels],
            "kind": kind,
            "type": config["type_list"][types[rel_i]],
            "degree": config["degree_list"][degrees[rel_i]],
            "relation_id": rel_i,
            "id": sample_id,
        }

        sample_res.append(pd.DataFrame(rel_result))

    return sample_res


def make_trigger_combs_labels(trigger_combs, trigger_groups):
    """Generates the labels for the trigger combination pairs.
    1 if both triggers are in the same group, 0 if they are not in a group.

    Parameters
    ----------
    trigger_combs : list
        Contains trigger combination pairs
    trigger_groups : list
        Contains trigger groups

    Returns
    -------
    list
        For each trigger combination, 1 if in same Trigger group, 0 if not
    """
    trigger_combs_labels = []
    for t1, t2 in trigger_combs: 
        found = -1
        for i, trigger_group in enumerate(trigger_groups):
            if t1 in trigger_group and t2 in trigger_group:
                found = i
        trigger_combs_labels.append(1. if found != -1 else 0.)
    
    return trigger_combs_labels


def make_trigger_groups(labels, subwords, config):
    """Generate the trigger groups from the 2d label tensor.
    Each row in this tensor denotes a relation, and the triggers of each 
    relation are grouped.
    Only use first B-token of word (ignore B-token in subsequent subwords)

    Parameters
    ----------
    labels : torch.tensor
        Tensor of all relations in a sample
    subwords : torch.tensor
        Tensor where subwords of tokens are numerated, 0 values are start
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    list
        Trigger groups of the sample, one group for each relation
    """
    trigger_rows, trigger_ind = torch.where((labels==config["label_dict"]["B-Trigger"]) & (subwords==0))

    trigger_groups = []
    for trigger_row in trigger_rows.unique():
        trigger_groups.append(trigger_ind[trigger_rows == trigger_row])

    return trigger_groups


def get_trigger_comb_input(trigger_groups):
    """Constructs training data for trigger groups.
    Returns all the combinations of Triggers, with labels indicating if they are
    in the same Trigger group.

    Parameters
    ----------
    trigger_groups : list
        List of tensors, each tensor contains the triggers of a relation

    Returns
    -------
    (torch.tensor, torch.tensor)
        Tensors containing the trigger combinations and the corresponding labels
    """

    if len(trigger_groups) == 0: return torch.tensor([]), torch.tensor([])

    unique_triggers = torch.unique(torch.cat(trigger_groups))
    trigger_combs = torch.combinations(unique_triggers, r=2)
    trigger_combs_labels = make_trigger_combs_labels(trigger_combs, trigger_groups)
    return trigger_combs, trigger_combs_labels


def filter_metrics_from_validation_results(results, config):
    """Filter full validation results to only keep important metrics.

    Parameters
    ----------
    results : dict
        Dictionary containing all tracked metrics
    config : dict
        Dictionary containing overall parameters and constants

    Returns
    -------
    pd.DataFrame
        DataFrame containing a subset of tracked metrics
    """

    for args in config["causal_arguments"]:
        if f"{args}_strict_f1" not in results["detect_args_results"]:
            results["detect_args_results"][f"{args}_strict_f1"] = 0
            results["detect_args_results"][f"{args}_relaxed_f1"] = 0

    filtered_results = pd.DataFrame({
            "Trigger F1": results["detect_trigger_results"]["Trigger_strict_f1"],
            "Combine Trigger macro F1": results["combine_trigger_results"]["f1_macro"],
            "Combine Trigger MCC": results["combine_trigger_results"]["MCC"],
            "Combine Trigger Accuracy": results["combine_trigger_results"]["accuracy"],
            "Combine Trigger % True": results["combine_trigger_results"]["% True"],
            "Detect Args F1 (strict)": results["detect_args_results"]["overall_f1_strict"],
            "Detect Args F1 (relaxed)": results["detect_args_results"]["overall_f1_relaxed"],
            "Detect Args Actor F1 (strict)": results["detect_args_results"]["Actor_strict_f1"],
            "Detect Args Actor F1 (relaxed)": results["detect_args_results"]["Actor_relaxed_f1"],
            "Detect Args Affected F1 (strict)": results["detect_args_results"]["Affected_strict_f1"],
            "Detect Args Affected F1 (relaxed)": results["detect_args_results"]["Affected_relaxed_f1"],
            "Detect Args Cause F1 (strict)": results["detect_args_results"]["Cause_strict_f1"],
            "Detect Args Cause F1 (relaxed)": results["detect_args_results"]["Cause_relaxed_f1"],
            "Detect Args Effect F1 (strict)": results["detect_args_results"]["Effect_strict_f1"],
            "Detect Args Effect F1 (relaxed)": results["detect_args_results"]["Effect_relaxed_f1"],
            "Detect Args Controlling F1 (strict)": results["detect_args_results"]["Controlling_strict_f1"],
            "Detect Args Controlling F1 (relaxed)": results["detect_args_results"]["Controlling_relaxed_f1"],
            "Detect Args Support F1 (strict)": results["detect_args_results"]["Support_strict_f1"],
            "Detect Args Support F1 (relaxed)": results["detect_args_results"]["Support_relaxed_f1"],
            "Classify Type MCC": results["classify_type_results"]["MCC"],
            "Classify Type macro F1": results["classify_type_results"]["f1_macro"],
            "Classify Type Accuracy": results["classify_type_results"]["accuracy"],
            "Classify Type Purpose F1": results["classify_type_results"]["purpose_f1"],
            "Classify Type Motivation F1": results["classify_type_results"]["motivation_f1"],
            "Classify Type Consequence F1": results["classify_type_results"]["consequence_f1"],
            "Classify Type IsCausal Accuracy": results["classify_type_results"]["is_causal_accuracy"],
            "Classify Type IsCausal %": results["classify_type_results"]["% causal"],
            "Classify Degree MCC": results["classify_degree_results"]["MCC"],
            "Classify Degree macro F1": results["classify_degree_results"]["f1_macro"],
            "Classify Degree Accuracy": results["classify_degree_results"]["accuracy"],
            "Classify Degree Facilitate %": results["classify_degree_results"]["% facilitate"],
            "Classify Degree Facilitate F1": results["classify_degree_results"]["facilitate_f1"],
            "Classify Degree Inihibit F1": results["classify_degree_results"]["inhibit_f1"],
        }, index=[0]
    )
    return filtered_results


def get_kfolds(sentence_data, config, n_splits=5, debug=False):
    """Split sentences into stratified kfolds based on type and degree.

    Parameters
    ----------
    sentence_data : list
        List of sentence samples with tokens and relations
    config : dict
        Dictionary containing overall parameters and constants
    n_splits : int, optional
        Number of folds, by default 5
    debug : bool, optional
        Whether debug output is provided, by default True

    Returns
    -------
    pd.DataFrame
        Contains the validation fold for each sample
    """

    sentence_ids = np.arange(len(sentence_data))
    stratification_data = []

    for sent_data in sentence_data:

        stratification_features_sent =  {}
        for relation in sent_data["relations"]:
            stratification_features_sent[relation["degree"]] = True
            stratification_features_sent[relation["type"]] = True

        stratification_data.append(stratification_features_sent)

    sentence_id_kfold = pd.DataFrame(stratification_data).fillna(False)
    sentence_id_kfold["sentence_id"] = sentence_ids
    sentence_id_kfold["fold"] = -1

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])
    labels = [c for c in sentence_id_kfold.columns if c not in ["sentence_ids", "fold"]]

    for fold, (_, test_ids) in enumerate(mskf.split(sentence_id_kfold, sentence_id_kfold[labels])):
        sentence_id_kfold.loc[test_ids, "fold"] = fold

    if debug:
        display(sentence_id_kfold.groupby("fold")[labels].sum())
    return sentence_id_kfold


def print_dataset_statistics(dataset, name, config):
    """Print information about the dataset.
    - Causal/non-causal
    - Number of arguments
    - Types
    - Degrees

    Parameters
    ----------
    dataset : torch.Dataset
        Dataset for statistics
    name : str
        Name of Dataset
    config : dict
        Dictionary containing overall parameters and constants
    """

    trigger_counts = pd.Series({"B-Trigger": 0})
    argument_counts = pd.Series({l: 0 for l in config["label_list"] if l != "O"})
    type_counts = pd.Series({t: 0 for t in config["type_list"]})
    degree_counts = pd.Series({t: 0 for t in config["degree_list"]})

    for i, sample in enumerate(dataset):

        sent_id = sample["sent_id"]
        tokens = dataset.__gettokens__(sent_id)
        relations = dataset.sentence_data[sent_id]["relations"]

        labels = np.array(get_labels_for_sentence(tokens, relations, dataset.add_coreferences))
        trigger_counts["B-Trigger"] += np.sum(np.sum(labels == "B-Trigger", axis=0) > 0)          

        for labels_rel in labels:
            is_trigger = labels_rel == "B-Trigger"
            for i, label in enumerate(labels_rel):
                if label == "O" or is_trigger[i]: continue
                argument_counts[label] += 1

        num_rels = (sample["sample_relation_types"] != -100).sum().item() # non-padded values
        for rel_i in range(num_rels):
            rel_type = config["type_list"][sample["sample_relation_types"][rel_i]]
            type_counts[rel_type] = type_counts[rel_type] + 1

            rel_degree = config["degree_list"][sample["sample_relation_degrees"][rel_i]]
            degree_counts[rel_degree] = degree_counts[rel_degree] + 1

    print(f"Statistics for Dataset {name}")
    print(pd.DataFrame(trigger_counts.astype(int)).T)
    print(pd.DataFrame(argument_counts.astype(int)).T)
    print(pd.DataFrame(type_counts.astype(int)).T)
    print(pd.DataFrame(degree_counts.astype(int)).T)


class NumpyEncoder(json.JSONEncoder):
    """
    Encoder for saving json object including numpy arrays
    From https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)