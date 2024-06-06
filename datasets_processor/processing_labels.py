#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Preprocessing before styleGAN2 training

import os
import sys

sys.path.append("../")
from glob import glob
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import torch
import re
import json

### ---------- Pathes and constants -------------

path_to_image_train = "/home/stympopper/data/DVMdata/features/train_paths_all_views.pt"
path_to_image_val = "/home/stympopper/data/DVMdata/features/val_paths_all_views.pt"
path_to_labels_train = "/home/stympopper/data/DVMdata/features/labels_model_all_train_all_views.pt"
path_to_labels_val = "/home/stympopper/data/DVMdata/features/labels_model_all_val_all_views.pt"
path_to_image = "/home/stympopper/data/DVMdata/features/"
path_to_json_train = "/home/stympopper/data/DVMdata/readyTrainLabeled"
path_to_json_val = "/home/stympopper/data/DVMdata/readyValLabeled"

### ---------- Functions -------------


def create_dict_label(path_to_image: str, path_to_labels: str) -> dict:
    list_paths = torch.load(path_to_image, map_location=torch.device('cpu'))
    list_labels = torch.load(path_to_labels, map_location=torch.device('cpu'))

    cnt = 0
    list_image = []
    for path in tqdm(list_paths, desc="Creating dict"):
        label = list_labels[cnt]
        list_image.append([path, label])
        cnt += 1

    label_dict = {}
    label_dict["labels"] = list_image

    return label_dict


def export_json(dict_label: dict, path: str) -> None:
    os.chdir(path)
    with open("dataset_val.json", "w") as f:
        json.dump(dict_label, f)

def export_pickle(dict_label: dict, path: str) -> None:
    os.chdir(path)
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dict_label, f)

def modify_json(path: str) -> None:
    with open(path + "dataset.json", "r") as f:
        data = json.load(f)["labels"]
    new_json = []
    new_json_dict = {}
    for path, labels in data:
        path = path.replace(".png", "_unaugmented.png")
        new_json.append([path, labels])
    
    with open(path + "dataset_unaugmented.json", "w") as f:
        json.dump({"labels": new_json}, f)


### ---------- Programs -------------

# if __name__ == "__main__":
#     dict_label = create_dict_label(path_to_image_val, path_to_labels_val)
#     export_json(dict_label, path_to_image)

if __name__ == "__main__":
    modify_json(path_to_json_train)
    modify_json(path_to_json_val)