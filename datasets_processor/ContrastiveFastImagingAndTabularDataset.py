from typing import List, Tuple
import random
import csv
import copy
import numpy as np

import torch
from torch.utils.data import Dataset, Subset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm

has_tqdm = True
import lmdb
import PIL

from .TabularAttributes import (
    check_categorical_data,
    CAT_FEATURES,
    NUM_FEATURES,
    CAT_FEATURES_WITH_LABEL,
)
from datasets_processor.ImageFastDataset import ImageFastDataset

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from multiprocessing.managers import ListProxy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast

import numpy as np
from torch.multiprocessing import Manager
from torch.serialization import DEFAULT_PROTOCOL


class ContrastiveFastImagingAndTabularDataset(Dataset):
    """
    Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

    The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
    The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
    with values chosen from the empirical marginal distribution of that feature.
    """

    def __init__(
        self,
        data_path_imaging: str,
        delete_segmentation: bool,
        augmentation_rate: float,
        data_path_tabular: str,
        corruption_rate: float,
        field_lengths_tabular: str,
        one_hot_tabular: bool,
        labels_path: str,
        img_size: int,
        target: str,
        missing_values: list = [],
        tabular_model: str = "mlp",
        use_labels: bool = False,
        max_size: int = None,
    ) -> None:

        # Imaging
        self.data_imaging_dataset = ImageFastDataset(
            data_path=data_path_imaging,
            name="imaging", 
            img_size=img_size,
            target=target,
            use_labels=use_labels,
            max_size=max_size,
            train_augment_rate=augmentation_rate,
            delete_segmentation=delete_segmentation,
        )

        self.delete_segmentation = delete_segmentation
        self._use_labels = use_labels

        if self.delete_segmentation:
            for im in self.data_imaging:
                im[0, :, :] = 0

        # Tabular
        use_header = True if tabular_model == "transformer" else False
        self.c = corruption_rate
        self.field_lengths_tabular = torch.load(field_lengths_tabular)
        self.one_hot_tabular = one_hot_tabular
        self.data_tabular = self.read_and_parse_csv(
            data_path_tabular, missing_values, use_header, max_size=max_size
        )
        self.generate_marginal_distributions(data_path_tabular)

        # Classifier
        self.labels = torch.load(labels_path)

        # Masking
        self.tabular_model = tabular_model

    def read_and_parse_csv(
        self,
        path_tabular: str,
        missing_values: list = [],
        use_header: bool = False,
        max_size: int = None,
    ) -> List[List[float]]:
        """
        Does what it says on the box.
        """
        if use_header and self._use_labels:
            FEATURES = NUM_FEATURES + CAT_FEATURES_WITH_LABEL
        elif use_header:
            FEATURES = NUM_FEATURES + CAT_FEATURES
            df = pd.read_csv(path_tabular, names=FEATURES)
            df.drop(missing_values, axis=0, inplace=True)
            cat_mask = check_categorical_data(df)
            self.cat_mask = cat_mask
            field_lengths_tensor = torch.as_tensor(self.field_lengths_tabular)
            self.cat_card = field_lengths_tensor[cat_mask]
            data = df.values.tolist()
        else:
            with open(path_tabular, "r") as f:
                reader = csv.reader(f)
                data = []
                if max_size is not None:
                    for idx, r in enumerate(reader):
                        if idx in missing_values:
                            continue
                        r2 = [float(r1) for r1 in r]
                        data.append(r2)
                        if idx >= max_size:
                            break
                else:
                    for idx, r in enumerate(reader):
                        if idx in missing_values:
                            continue
                        r2 = [float(r1) for r1 in r]
                        data.append(r2)
        return data

    def generate_marginal_distributions(self, data_path: str) -> None:
        """
        Generates empirical marginal distribution by transposing data
        """
        data_df = pd.read_csv(data_path)
        self.marginal_distributions = data_df.transpose().values.tolist()

    def get_input_size(self) -> int:
        """
        Returns the number of fields in the table.
        Used to set the input number of nodes in the MLP
        """
        if self.one_hot_tabular:
            return int(sum(self.field_lengths_tabular))
        else:
            return len(self.data_tabular[0])

    def corrupt(self, subject: List[float]) -> List[float]:
        """
        Creates a copy of a subject, selects the indices
        to be corrupted (determined by hyperparam corruption_rate)
        and replaces their values with ones sampled from marginal distribution
        """
        subject = deepcopy(subject)

        indices = random.sample(list(range(len(subject))), int(len(subject) * self.c))
        for i in indices:
            subject[i] = random.sample(self.marginal_distributions[i], k=1)[0]
        return subject

    def create_mask(self, subject: List[float], use_mask: bool = True) -> List[bool]:
        """
        Creates a mask of features to be corrupted
        """
        if not use_mask:
            return None
        mask = [False] * len(subject)
        random.seed(42)
        indices = random.sample(list(range(len(subject))), int(len(subject) * self.c))
        for i in indices:
            mask[i] = True
        return torch.as_tensor(mask)

    def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
        """
        One-hot encodes a subject's features
        """
        out = []
        for i in range(len(subject)):
            if self.field_lengths_tabular[i] == 1:
                out.append(subject[i].unsqueeze(0))
            else:
                out.append(
                    torch.nn.functional.one_hot(
                        subject[i].long(),
                        num_classes=int(self.field_lengths_tabular[i]),
                    )
                )
        return torch.cat(out)

    def get_label(self, index: int) -> int:
        """
        Returns the label of the subject at index
        """
        label = self.data_imaging_dataset.get_label(index)
        assert label == self.labels[index], f"{label} != {self.labels[index]}"
        return torch.as_tensor(label, dtype=torch.long)

    def get_cat_mask(self) -> torch.Tensor:
        """
        Returns the categorical mask
        """
        return torch.as_tensor(self.cat_mask)

    def get_cat_card(self) -> torch.Tensor:
        """
        Returns the categorical cardinalities
        """
        return torch.as_tensor(self.cat_card)

    def get_number_of_numerical_features(self) -> int:
        """
        Returns the number of numerical features
        """
        return len(NUM_FEATURES)

    def __getitem__(
        self, index: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        imaging_views = self.data_imaging_dataset.get_image_from_idx(index)
        tabular_views = torch.tensor(self.data_tabular[index])
        tabular_mask = self.create_mask(self.data_tabular[index])
        label = self.get_label(index)
        return (imaging_views, tabular_views, tabular_mask), label

    def __len__(self) -> int:
        return len(self.data_tabular)
