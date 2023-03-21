# -*- coding: utf-8 -*-

from transformers import BertTokenizer
from role_semantic import role_semantic, event_role, all_roles
import numpy as np
import json
import os
from load_data import load_data
import pandas as pd
from tqdm import tqdm
import networkx as nx
import spacy

nlp = spacy.load("en_core_web_sm")

from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser


# tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

ent_label_words = {
    "FAC": "facility",
    "ORG": "organization",
    "GPE": "geographical or political entity",
    "PER": "person",
    "VEH": "vehicle",
    "WEA": "weapon",
    "LOC": "location",
    "ART": "artifact",
    "TIME": "time",
}

event_label_words = {
    "Movement:Transport": "transport",
    "Personnel:Elect": "election",
    "Personnel:Start-Position": "employment",
    "Personnel:Nominate": "nomination",
    "Personnel:End-Position": "dimission",
    "Conflict:Attack": "attack",
    "Contact:Meet": "meeting",
    "Life:Marry": "marriage",
    "Transaction:Transfer-Money": "money transfer",
    "Conflict:Demonstrate": "demonstration",
    "Business:End-Org": "collapse",
    "Justice:Sue": "prosecution",
    "Life:Injure": "injury",
    "Life:Die": "death",
    "Justice:Arrest-Jail": "arrest or jail",
    "Contact:Phone-Write": "written or telephone communication",
    "Transaction:Transfer-Ownership": "ownership transfer",
    "Business:Start-Org": "organization founding",
    "Justice:Execute": "execution",
    "Justice:Trial-Hearing": "trial or hearing",
    "Life:Be-Born": "birth",
    "Justice:Charge-Indict": "charge or indict",
    "Justice:Convict": "conviction",
    "Justice:Sentence": "sentence",
    "Business:Declare-Bankruptcy": "bankruptcy",
    "Justice:Release-Parole": "release or parole",
    "Justice:Fine": "fine",
    "Justice:Pardon": "pardon",
    "Justice:Appeal": "appeal",
    "Justice:Extradite": "extradition",
    "Life:Divorce": "divorce",
    "Business:Merge-Org": "organization merger",
    "Justice:Acquit": "acquittal",
}

ent_label = ent_label_words.keys()
event_label = event_label_words.keys()
add_tokens = []  # add ID


label_word = list(role_semantic.keys())


class_label = {}
Token_id = []
i = 0
for word in label_word:
    class_label[word] = i
    i = i + 1
    # word_id = tokenizer.convert_tokens_to_ids(word)
    # Token_id.append(word_id)

event_role_id = {}
none_id = class_label["None"]
place_id = class_label["Place"]
time_id = class_label["Time"]


def remove_overlap_entities(entities):
    """There are a few overlapping entities in the data set. We only keep the
    first one and map others to it.
    :param entities (list): a list of entity mentions.
    :return: processed entity mentions and a table of mapped IDs.
    """
    tokens = [None] * 1000
    entities_ = []
    id_map = {}
    for entity in entities:
        start, end = entity["start"], entity["end"]
        for i in range(start, end):
            if tokens[i]:
                id_map[entity["id"]] = tokens[i]
                continue
        entities_.append(entity)
        for i in range(start, end):
            tokens[i] = entity["id"]
    return entities_, id_map


def data_to_proto_re(file_list, mode):
    """Convert the data to the format of ProtoRE.
    :param file_list (list): a list of file names.
    :param mode (str): "train" or "test".
    :return: a list of FewRel samples.
    """
    # TODO:
    # 1. process only gold arguments mentions
    # 2. process gold entities (add None role)
    # 3. Use candidat entities from anthoer model

    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
        ]
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
        ]

    inst_list = []

    relation_list = []

    for inst in tqdm(file_list, total=len(file_list), desc="Loading data"):
        sent_id = inst["sent_id"]
        inst_id = 0

        tokens = ["[CLS]"] + inst["pieces"].copy() + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [str(id) for id in input_ids]

        input_ids = "\002".join(input_ids)

        entities = inst["entity_mentions"]

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        events = inst["event_mentions"]
        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            # Trigger
            trig_start = str(event["trigger"]["start"])
            # trig_end = event["trigger"]["end"]
            # trig_text = event["trigger"]["text"]

            event_type = event["event_type"]

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            for cur_ent_id in ent_ids:
                (ent_start, ent_end, ent_text, ent_type) = ent_dict[cur_ent_id]

                ent_start = str(ent_start)

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if "Time" in ent_role:
                        ent_role = "Time"
                    relation_list.append(ent_role)

                    if event_type in types:
                        rel_id = sent_id + "_" + str(inst_id)

                        instance = "\001".join(
                            [rel_id, ent_role, trig_start, ent_start, input_ids]
                        )

                        inst_list.append(instance)
                        inst_id += 1

                """
                else:
                    ent_role = "None"
                """

    output_dir = "./data/ProtoRE_data/"

    print(len(inst_list))
    f = open(os.path.join(output_dir, f"meta_{mode}_args.txt"), "w")
    for inst in inst_list:
        f.write(inst + "\n")
    f.close()

    relation_list = list(set(relation_list))
    print(f"{len(relation_list)} relations")
    print(relation_list)
    print("Done")
    return inst_list


def data_to_few_rel(file_list, mode):
    """Convert the data to the format of FewRel.
    :param file_list (list): a list of file names.
    :param mode (str): "train" or "test".
    :return: {Event type: {role1: [instances, ...], ...}, ...}
    """

    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Business:End-Org",
            "Business:Declare-Bankruptcy",
            "Business:Merge-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
            "Justice:Pardon",
            "Justice:Appeal",
            "Justice:Acquit",
            "Justice:Extradite",
            "Justice:Fine",
            "Justice:Sue",
            "Justice:Release-Parole",
        ]  # 21 event types
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Marry",
            "Life:Divorce",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
            "Personnel:Nominate",
            "Transaction:Transfer-Money",
        ]  # 12 event types

    new_data_event = {
        event_type: {role: [] for role in event_role[event_type]}
        for event_type in event_label
        if event_type in types
    }

    for inst in file_list:
        sent = inst["tokens"]
        entities = inst["entity_mentions"]
        entities, ent_map = remove_overlap_entities(entities)

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}  # key:ID，value: (start, end, text, ent-type)
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        events = inst["event_mentions"]
        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            # Trigger
            trig_start = event["trigger"]["start"]
            trig_end = event["trigger"]["end"]
            trig_text = event["trigger"]["text"]

            event_type = event["event_type"]  # Event type
            event_id = event["id"]  # Event ID

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            for cur_ent_id in ent_ids:
                (ent_start, ent_end, ent_text, _) = ent_dict[cur_ent_id]
                ent_text = ent_text[0]

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if "Time" in ent_role:
                        ent_role = "Time"
                else:
                    ent_role = "None"

                if event_type in types:
                    if ent_role in new_data_event[event_type]:
                        new_data_event[event_type][ent_role].append(
                            {
                                "tokens": sent,
                                "h": [
                                    trig_text,
                                    event_id,
                                    [[i for i in range(trig_start, trig_end + 1)]],
                                ],
                                "t": [
                                    ent_text,
                                    cur_ent_id,
                                    [[i for i in range(ent_start, ent_end + 1)]],
                                ],
                            }
                        )
                    """
                    else:
                        new_data_event[event_type][ent_role] = []

                        new_data_event[event_type][ent_role].append(
                            {
                                "tokens": sent,
                                "h": [
                                    trig_text,
                                    event_id,
                                    [[i for i in range(trig_start, trig_end + 1)]],
                                ],
                                "t": [
                                    ent_text,
                                    cur_ent_id,
                                    [[i for i in range(ent_start, ent_end + 1)]],
                                ],
                            }
                        )
                    """
    # filter out roles with less than 6 instances
    final_data_event = {}
    for event in new_data_event:
        for role in new_data_event[event]:
            if len(new_data_event[event][role]) < 6:
                print(event, role, len(new_data_event[event][role]))
            else:
                if event not in final_data_event:
                    final_data_event[event] = {}
                final_data_event[event][role] = new_data_event[event][role]

    out = open("./out.log", "a")
    print(mode, file=out)
    for event in new_data_event:
        print(event, len(new_data_event[event]), "arguments", file=out)
        for role in new_data_event[event]:
            print(role, len(new_data_event[event][role]), file=out)
        print(file=out)
    out.close()

    output_dir = "./data/fewrel_type_none/"

    with open(os.path.join(output_dir, f"{mode}.few.rel.json"), "w") as f:
        json.dump(final_data_event, f)
    f.close()

    return final_data_event


def data_to_few_rel_v2(file_list, mode):
    """
    Convert the data to fewrel format (event_type level) - test/dev
    """

    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Business:End-Org",
            "Business:Declare-Bankruptcy",
            "Business:Merge-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
            "Justice:Pardon",
            "Justice:Appeal",
            "Justice:Acquit",
            "Justice:Extradite",
            "Justice:Fine",
            "Justice:Sue",
            "Justice:Release-Parole",
        ]  # 21 event types
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Marry",
            "Life:Divorce",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
            "Personnel:Nominate",
            "Transaction:Transfer-Money",
        ]  # 12 event types

    new_data_event = {
        event_type: [] for event_type in event_label if event_type in types
    }

    for inst in file_list:
        tokens = inst["tokens"].copy()
        entities = inst["entity_mentions"]
        sent = inst["sentence"]
        print(sent)
        deps_parse = dependency_parser.raw_parse(sent)
        deps = next(deps_parse)
        deps = deps.nodes.values()

        entities, _ = remove_overlap_entities(entities)

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}  # key:ID，value: (start, end, text, ent-type)
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        events = inst["event_mentions"]

        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            inst_args = []
            # Trigger
            trigger = event["trigger"]
            trigger["id"] = event["id"]
            event_type = event["event_type"]

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            for cur_ent_id in ent_ids:
                (ent_start, ent_end, ent_text, ent_type) = ent_dict[cur_ent_id]
                ent_text = ent_text[0]

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if "Time" in ent_role:
                        ent_role = "Time"
                else:
                    ent_role = "None"

                tri_start = trigger["start"]

                print("Trigger:", trigger["text"], tri_start, trigger["end"])
                print("Entity:", ent_text, ent_start, ent_end)

                print("Role:", ent_role)
                print("Dep:", list(deps)[ent_start])
                print("-" * 50)
                # tok = " ".join(tokens)
                # dep = find_dep(tri_start, ent_start, tok)
                # print("Dep:", dep)

                inst_args.append(
                    {
                        "text": ent_text,
                        "id": cur_ent_id,
                        "start": ent_start,
                        "end": ent_end,
                        "role": ent_role,
                        "type": ent_type,
                    }
                )
                if event_type in types:
                    new_data_event[event_type].append(
                        {
                            "tokens": tokens,
                            "arguments": inst_args,
                            "trigger": trigger,
                        }
                    )
    final_data_event = new_data_event.copy()
    for event in new_data_event:
        args = {}
        for inst in new_data_event[event]:
            for arg in inst["arguments"]:
                if arg["role"] not in args:
                    args[arg["role"]] = 1
                else:
                    args[arg["role"]] += 1
        print(event, args)

        # remove roles with less than 6 instances
        for inst in new_data_event[event]:
            new_args = []
            for arg in inst["arguments"]:
                if args[arg["role"]] >= 6:
                    new_args.append(arg)
                else:
                    print("remove", arg["role"], args[arg["role"]])
            inst["arguments"] = new_args
        # remove events with no arguments left after filtering
        # if len(new_data_event[event]) == 0:
        #    del new_data_event[event]
        print(event)
        print(len(new_data_event[event]))
        print("-" * 20)

    output_dir = "./data/fewrel_sentence/"

    with open(os.path.join(output_dir, f"{mode}.few.rel.sentence.json"), "w") as f:
        json.dump(new_data_event, f)
    f.close()

    return new_data_event


def data_to_few_rel_v3(file_list, mode):
    """
    Rôle level only for train with Nones
    """

    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Business:End-Org",
            "Business:Declare-Bankruptcy",
            "Business:Merge-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
            "Justice:Pardon",
            "Justice:Appeal",
            "Justice:Acquit",
            "Justice:Extradite",
            "Justice:Fine",
            "Justice:Sue",
            "Justice:Release-Parole",
        ]  # 21 event types
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Marry",
            "Life:Divorce",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
            "Personnel:Nominate",
            "Transaction:Transfer-Money",
        ]  # 12 event types

    new_data_event = {role: [] for role in all_roles}
    print(f"{len(new_data_event)} roles")
    print(new_data_event.keys())

    for inst in file_list:
        sent = inst["tokens"].copy()

        entities = inst["entity_mentions"]

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        events = inst["event_mentions"]
        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            # Trigger
            trig_start = event["trigger"]["start"]
            trig_end = event["trigger"]["end"]
            trig_text = event["trigger"]["text"]

            event_type = event["event_type"]  # Event type
            event_id = event["id"]  # Event ID

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            for cur_ent_id in ent_ids:
                (ent_start, ent_end, ent_text, _) = ent_dict[cur_ent_id]
                ent_text = ent_text[0]

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if "Time" in ent_role:
                        ent_role = "Time"
                    else:
                        ent_role = "None"

                    if event_type in types:
                        new_data_event[ent_role].append(
                            {
                                "tokens": sent,
                                "h": [
                                    trig_text,
                                    event_id,
                                    [[i for i in range(trig_start, trig_end + 1)]],
                                ],
                                "t": [
                                    ent_text,
                                    cur_ent_id,
                                    [[i for i in range(ent_start, ent_end + 1)]],
                                ],
                            }
                        )

    # filter out roles with less than 10 instances
    final_data_event = {}

    for role in new_data_event.keys():
        if len(new_data_event[role]) < 7:
            print(event, role, len(new_data_event[role]))
        else:
            if role not in final_data_event:
                final_data_event[role] = []
            final_data_event[role] = new_data_event[role]

    output_dir = "./data/fewrel_role_none"

    with open(os.path.join(output_dir, f"meta_{mode}_fewrel.json"), "w") as f:
        json.dump(final_data_event, f)
    f.close()

    return final_data_event


def data_to_few_rel_v4(file_list, mode):
    """
    Rôle level for only train
    """

    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Business:End-Org",
            "Business:Declare-Bankruptcy",
            "Business:Merge-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
            "Justice:Pardon",
            "Justice:Appeal",
            "Justice:Acquit",
            "Justice:Extradite",
            "Justice:Fine",
            "Justice:Sue",
            "Justice:Release-Parole",
        ]  # 21 event types
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Marry",
            "Life:Divorce",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
            "Personnel:Nominate",
            "Transaction:Transfer-Money",
        ]  # 12 event types

    new_data_event = {role: [] for role in all_roles}
    print(f"{len(new_data_event)} roles")
    print(new_data_event.keys())

    for inst in file_list:
        sent = inst["tokens"].copy()

        entities = inst["entity_mentions"]

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        events = inst["event_mentions"]
        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            # Trigger
            trig_start = event["trigger"]["start"]
            trig_end = event["trigger"]["end"]
            trig_text = event["trigger"]["text"]

            event_type = event["event_type"]  # Event type
            event_id = event["id"]  # Event ID

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            for cur_ent_id in ent_ids:
                (ent_start, ent_end, ent_text, _) = ent_dict[cur_ent_id]
                ent_text = ent_text[0]

                instance = {
                    "sent-id": None,
                    "tokens": None,
                    "h": None,
                    "args": [],
                }

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if "Time" in ent_role:
                        ent_role = "Time"
                    else:
                        ent_role = "None"

                    instance["sent-id"] = inst["sent_id"]
                    instance["tokens"] = sent
                    instance["h"] = [trig_text, event_id, [[trig_start, trig_end]]]
                    instance["args"].append(
                        [ent_text, cur_ent_id, [[ent_start, ent_end]], ent_role]
                    )
                    if event_type in types:
                        new_data_event[event_type].append(instance)

    # filter out roles with less than 10 instances
    final_data_event = {}

    for role in new_data_event.keys():
        if len(new_data_event[role]) < 10:
            print(event, role, len(new_data_event[role]))
        else:
            if role not in final_data_event:
                final_data_event[role] = []
            final_data_event[role] = new_data_event[role]

    output_dir = "./data/fewrel_role"

    with open(os.path.join(output_dir, f"meta_{mode}_fewrel.json"), "w") as f:
        json.dump(final_data_event, f)
    f.close()

    return final_data_event


def data_stats(file_list, mode):
    if mode == "train":
        types = [
            "Conflict:Attack",
            "Contact:Meet",
            "Conflict:Demonstrate",
            "Justice:Arrest-Jail",
            "Contact:Phone-Write",
            "Business:Start-Org",
            "Justice:Execute",
            "Justice:Trial-Hearing",
            "Justice:Charge-Indict",
            "Justice:Convict",
            "Justice:Sentence",
        ]
    else:
        types = [
            "Movement:Transport",
            "Personnel:Elect",
            "Personnel:Start-Position",
            "Personnel:End-Position",
            "Life:Injure",
            "Life:Die",
            "Transaction:Transfer-Ownership",
            "Life:Be-Born",
        ]

    new_data_event = {
        event_type: {} for event_type in event_label if event_type in types
    }

    new_data_role = []

    event_labels = []

    input_sents = []
    input_sents_inv = []
    arg_labels = []
    length = []
    arg_num = 0
    label_masks = []

    for inst in file_list:
        # tokens = tokenizer.tokenize(tokens)

        entities = inst["entity_mentions"]

        entities.sort(key=lambda x: x["start"])
        ent_dict = {}  # key:ID，value: (start, end, text, ent-type)
        ent_ids = []

        for entity in entities:
            ent_dict[entity["id"]] = (
                entity["start"],
                entity["end"],
                entity["text"],
                entity["entity_type"],
            )
            ent_ids.append(entity["id"])

        ent_ids_inv = ent_ids.copy()
        ent_ids_inv.reverse()
        events = inst["event_mentions"]
        events.sort(key=lambda x: x["trigger"]["start"])

        for event in events:
            # Trigger
            trig_start = event["trigger"]["start"]
            trig_end = event["trigger"]["end"]
            trig_text = event["trigger"]["text"]

            event_type = event["event_type"]  # Event type

            # modif ici !
            label_mask = [1] * len(class_label)
            label_mask = np.array(label_mask)
            label_mask[event_role_id[event_type]] = [0] * len(event_role_id[event_type])
            label_mask = label_mask.tolist()

            # args
            arguments = event["arguments"]
            arg_dict = {}

            for arg in arguments:
                arg_dict[arg["entity_id"]] = arg["role"]

            sent = inst["tokens"]

            for cur_ent_id in ent_ids:
                Tokens = []
                triggers = []
                entities_list = []
                role_labels = []

                (ent_start, ent_end, ent_text, ent_type) = ent_dict[cur_ent_id]

                new_ent_text = " or ".join(ent_text)

                if cur_ent_id in arg_dict.keys():
                    ent_role = arg_dict[cur_ent_id]
                    if ent_role.startswith("Time"):
                        ent_role = "Time"
                else:
                    ent_role = "None"
                if ent_role not in new_data_event[event_type]:
                    new_data_event[event_type][ent_role] = []

                Tokens.append(sent)
                triggers.append(
                    {"text": trig_text, "start": trig_start, "end": trig_end}
                )

                entities_list.append(
                    {
                        "text": new_ent_text,
                        "start": ent_start,
                        "end": ent_end,
                        "type": ent_type,
                    }
                )
                role_labels.append(ent_role)

                new_role_data = {
                    ent_role: [
                        {
                            "tokens": Tokens[i],
                            "trigger": triggers[i],
                            "entity": entities_list[i],
                        }
                        for i in range(len(triggers))
                    ]
                }
                print(set(role_labels))
                if event_type in types:
                    for i, role in enumerate(new_data_event[event_type].keys()):
                        if role in set(role_labels):
                            for new_inst in new_role_data[role]:
                                new_data_event[event_type][role].append(new_inst)

    df = pd.DataFrame(columns=["Event-type", "Role", "Role-count"])

    for event in new_data_event:
        for role in new_data_event[event]:
            df.loc[len(df)] = [event, role, len(new_data_event[event][role])]

    df.to_csv(f"./data/output_{mode}.csv", index=False)

    return df


nlp = spacy.load("en_core_web_sm")

dep_list = {
    "ROOT": "root",
    "acl": "Clausal modifier of noun",
    "acomp": "Adjectival complement",
    "advcl": "Adverbial clause modifier",
    "advmod": "Adverbial modifier",
    "agent": "Agent",
    "amod": "Adjectival modifier",
    "appos": "Appositional modifier",
    "attr": "Attribute",
    "aux": "Auxiliary",
    "auxpass": "Passive auxiliary",
    "case": "Case marking",
    "cc": "Coordinating conjunction",
    "ccomp": "Clausal complement",
    "compound": "Compound modifier",
    "conj": "Conjunct",
    "csubj": "Clausal subject",
    "csubjpass": "Clausal passive subject",
    "dative": "Dative",
    "dep": "Unspecified dependency",
    "det": "Determiner",
    "dobj": "Direct object",
    "expl": "Expletive",
    "intj": "Interjection",
    "mark": "Marker",
    "meta": "Metadata",
    "neg": "Negation modifier",
    "nmod": "Modifier of nominal",
    "npadvmod": "Noun phrase as adverbial modifier",
    "nsubj": "Nominal subject",
    "nsubjpass": "Passive nominal subject",
    "nummod": "Numeric modifier",
    "oprd": "Object predicate",
    "parataxis": "Parataxis",
    "pcomp": "The complement of a preposition",
    "pobj": "Object of a preposition",
    "poss": "Possession modifier",
    "preconj": "Pre-correlative conjunction",
    "predet": "Predeterminer",
    "prep": "Prepositional modifier",
    "prt": "Particle",
    "punct": "Punctuation",
    "quantmod": "Modifier of quantifier",
    "relcl": "Relative clause modifier",
    "xcomp": "Open clausal complement",
}


def find_head(arg_start, arg_end, doc):
    arg_end -= 1
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <= arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head
            break
        else:
            cur_i = doc[cur_i].head.i

    arg_head = cur_i
    head_text = doc[arg_head]
    return head_text


def shortest_dependency_path(doc, e1=None, e2=None):
    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token, child))
    graph = nx.Graph(edges)
    try:
        shortest_path = nx.shortest_path(graph, source=e1, target=e2)
    except nx.NetworkXNoPath:
        shortest_path = []

    # remove non significant dependencies
    aux = [
        "det",
        "prep",
        "pobj",
        "punct",
        "amod",
        "dep",
        "aux",
        "auxpass",
        "case",
        "quantmod",
        "compound",
    ]
    try:
        shortest_path = [i for i in shortest_path if i.dep_ not in aux]
    except:
        pass

    if len(shortest_path) == 2:
        print("Direct dependency")
        relation = shortest_path[1].dep_
        print("Relation: ", relation)

    elif len(shortest_path) > 2:
        print("Indirect dependency")
        relation = [shortest_path[-i].dep_ for i in range(1, len(shortest_path))]
        print("Relation: ", relation)

    else:
        print("No dependency")
        relation = ["None"]
    return relation


def find_dep(pos1, pos2, sent):

    print("Sentence: ", sent)
    doc = nlp(sent)
    # doc = spacy.tokens.doc.Doc(
    #    nlp.vocab, words=sent, spaces=[True] * (len(sent) - 1) + [False]
    # )
    wordid1 = pos1[0]
    wordid2 = pos2[0]
    head = find_head(pos2[0], pos2[1], doc)
    print("Head entity: ", head)

    word1 = doc[wordid1]
    word2 = doc[wordid2]
    print("Word1: ", word1)
    print("Word2: ", word2)

    return shortest_dependency_path(doc, e1=word1, e2=word2)


if __name__ == "__main__":
    # Load the data
    input_dir = "./data/input_time"

    train_data = load_data("./data/input_time/train.fewshot.json")
    dev_data = load_data("./data/input_time/dev.fewshot.json")
    test_data = load_data("./data/input_time/test.fewshot.json")

    """
    new_train = data_stats(train_data, mode='train')
    print(new_train.head(10))
    ax = sns.barplot(x="Event-type", y="Role-count", hue="Role", data=new_train, orient='v')
    plt.savefig('./data/train.png')
    new_dev = data_stats(dev_data, mode='dev')
    new_test = data_stats(test_data, mode='test')
    
    """
    new_train = data_to_few_rel_v2(train_data, mode="train")
    new_dev = data_to_few_rel_v2(dev_data, mode="dev")
    new_test = data_to_few_rel_v2(test_data, mode="test")
