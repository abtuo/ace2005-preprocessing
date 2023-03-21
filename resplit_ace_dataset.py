#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:04:42 2021

@author: abou
"""
# imports
import pandas as pd
from tqdm import tqdm
import numpy as np
import random

import os

import json


train_types = ["Business", "Contact", "Conflict", "Justice"]
test_types = ["Life", "Movement", "Personnel", "Transaction"]

# random splits
random_split = False
if random_split:
    train_types = random.sample(train_types + test_types, 4)
    test_types = [t for t in train_types + test_types if t not in train_types]
    print(train_types, test_types, sep="\n")


def split_oneie_for_few_shot(path):
    train_data, test_data = [], []

    with open(path, "r", encoding="utf-8") as f:
        r = json.load(f)

        for line in r:
            inst = line
            if len(inst["golden-event-mentions"]):
                for event in inst["golden-event-mentions"]:
                    if event["event_type"].split(":")[0] in train_types:
                        train_data.append(line)
                    else:
                        test_data.append(line)
    f.close()

    return train_data, test_data


def split_ace(paths):
    """
    Parameters
    ----------
    paths : List of train/dev/dev .json files
        DESCRIPTION.

    Returns
    -------
    train.json : dict
        dict with keys=event-types and values=sentences
    """
    train_data, test_data = [], []

    for path in paths:
        train_data += split_oneie_for_few_shot(path)[0]
        test_data += split_oneie_for_few_shot(path)[1]

    random.shuffle(test_data)

    final_dev_data, final_test_data = (
        test_data[: len(test_data) // 2],
        test_data[len(test_data) // 2 :],
    )

    output_dir = "./data/input_time/"

    with open(os.path.join(output_dir, "train.fewshot.json"), "w") as w_train, open(
        os.path.join(output_dir, "dev.fewshot.json"), "w"
    ) as w_dev, open(os.path.join(output_dir, "test.fewshot.json"), "w") as w_test:
        #for line in train_data:
        json.dump(train_data, w_train, indent=4)
            #w_train.write(line)
        #for line in final_dev_data:
        json.dump(final_dev_data, w_dev, indent=4)
            #w_dev.write(line)
        #for line in final_test_data:
        json.dump(final_test_data, w_test, indent=4)
            #w_test.write(line)

    w_train.close()
    w_dev.close()
    w_test.close()


if __name__ == "__main__":
    root = "./output/"
    paths = [root + p for p in ["train.json", "dev.json", "test.json"]]

    split_ace(paths)
