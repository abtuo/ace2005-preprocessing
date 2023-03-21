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


def split_oneie_for_few_shot(path):
    train_data, test_data = [], []

    # split the dataset

    with open(path, "r", encoding="utf-8") as r:
        for line in r:
            inst = json.loads(line)
            if len(inst["event_mentions"]):
                for event in inst["event_mentions"]:
                    if event["event_type"].split(":")[0] in train_types:
                        train_data.append(line)
                    else:
                        test_data.append(line)
    r.close()

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
        for line in train_data:
            w_train.write(line)
        for line in final_dev_data:
            w_dev.write(line)
        for line in final_test_data:
            w_test.write(line)

    w_train.close()
    w_dev.close()
    w_test.close()


if __name__ == "__main__":
    root = "./data/input_time/"
    paths = [
        root + p for p in ["train.oneie.json", "dev.oneie.json", "test.oneie.json"]
    ]

    split_ace(paths)
