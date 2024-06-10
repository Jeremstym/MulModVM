from typing import Tuple
import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm
import numpy as np
import json

has_tqdm = True
import PIL

from utils.utils import (
    grab_hard_eval_image_augmentations,
    grab_soft_eval_image_augmentations,
    grab_image_augmentations,
)
from .TabularAttributes import (
    check_categorical_data,
    CAT_FEATURES,
    NUM_FEATURES,
    CAT_FEATURES_WITH_LABEL,
)

import collections.abc
import math
import pickle
import shutil
import sys
import os
import tempfile
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from copy import copy, deepcopy
from multiprocessing.managers import ListProxy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, cast, List, Optional, Tuple, Union

try:
    import pyspng
except ImportError:
    pyspng = None


class ImageFastDataset(Dataset):
    """
    Dataset class for images, after processed them with LMDB (from StyleGAN2-ADA repository, dataset_tools).
    """

    def __init__(
        self,
        data_path: str,
        name: str,
        use_labels: bool = False,
        train_augment_rate: float = 0.0,
        # use_augmented: bool = False,
        train: bool = True,
        delete_segmentation: bool = False,
        max_size: int = None,
        resolution: int = None,
        one_hot_labels: bool = False,
        labels_path_short: str = None,
        **super_kwargs,
    ):
        self._path = data_path
        self._name = name
        self._use_labels = use_labels
        self._train_augment_rate = train_augment_rate
        # self._use_augmented = use_augmented
        self._raw_labels = None
        self._label_shape = None
        self.one_hot_labels = one_hot_labels
        self._label_path = labels_path_short
        self.train = train

        self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
        self.transform_val = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(img_size,img_size)),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x : x.float())
        ])
        assert os.path.isdir(self._path), f"Path {self._path} is not a directory"
        self._type = "dir"
        self._all_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self._path)
            for root, _dirs, files in os.walk(self._path)
            for fname in files
        }
        # if not self._use_augmented:
        #     self._path = data_path + "/unaugmented"
        #     assert os.path.isdir(self._path), f"Path {self._path} is not a directory"
        #     self._type = "dir"
        #     self._all_fnames = {
        #         os.path.relpath(os.path.join(root, fname), start=self._path)
        #         for root, _dirs, files in os.walk(self._path)
        #         for fname in files
        #     }
        # elif self._use_augmented and (self._train_augment_rate > 0.0): #TODO: make a mix of augmented and unaugmented
        #     self._path1 = data_path + "/augmented"
        #     assert os.path.isdir(self._path1), f"Path {self._path1} is not a directory"
        #     self._path2 = data_path + "/unaugmented"
        #     assert os.path.isdir(self._path2), f"Path {self._path2} is not a directory"
        #     self._paths = [self._path1, self._path2]     
        #     self._type = "dir"
        #     self._all_fnames1 = {
        #         os.path.relpath(os.path.join(root, fname), start=self._path1)
        #         for root, _dirs, files in os.walk(self._path1)
        #         for fname in files
        #     }
        #     self._all_fnames2 = {
        #         os.path.relpath(os.path.join(root, fname), start=self._path2)
        #         for root, _dirs, files in os.walk(self._path2)
        #         for fname in files
        #     }
        # else:
        #     self._path = data_path + "/augmented"
        #     assert os.path.isdir(self._path), f"Path {self._path} is not a directory"
        #     self._type = "dir"
        #     self._all_fnames = {
        #         os.path.relpath(os.path.join(root, fname), start=self._path)
        #         for root, _dirs, files in os.walk(self._path)
        #         for fname in files
        #     }


        PIL.Image.init()
        self._image_fnames = sorted(
            fname
            for fname in self._all_fnames
            if self._file_ext(fname) in PIL.Image.EXTENSION
        )
        if len(self._image_fnames) == 0:
            raise IOError("No image files found in the specified path")

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        self._raw_shape = raw_shape
        if resolution is not None and (
            raw_shape[2] != resolution or raw_shape[3] != resolution
        ):
            raise IOError("Image files do not match the specified resolution")

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname, parent: bool = False):
        if self._type == "dir":
            # if parent:
            #     parent = os.path.dirname(self._path)
            #     return open(os.path.join(parent, fname), "rb")
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == ".png":
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        # image = image.transpose(2, 0, 1)  # HWC => CHW
        image = torch.from_numpy(image).float()
        if self.train and random.random() <= self._train_augment_rate:
            image = self.transform_train(image)
        else:
            image = self.transform_val(image)
        return image

    def get_image_from_idx(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        image = image.astype(np.float32)
        return image

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def _load_raw_labels(self):
        # if not self._use_augmented:
        #     fname = 'dataset_unaugmented.json'
        # else:
            # fname = 'dataset.json'
        fname = 'dataset.json'
        # parent = os.path.dirname(self._path)
        # if "dataset.json" not in os.listdir(parent):
        if fname not in self._all_fnames:
            print(f'WARNING: dataset is missing labels ({"dataset.json"})')
            return None
        # if fname not in self._all_fnames:
        #     print(f'WARNING: dataset is missing labels ({fname})')
        #     return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_label(self, idx):
        if self._label_path is not None:
            label = torch.load(self._label_path)[self._raw_idx[idx]]
            return label
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            if self.one_hot_labels:
                onehot = np.zeros(self.label_shape, dtype=np.float32)
                onehot[label] = 1
                label = onehot
        return label.copy()

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        image = image.astype(np.float32)
        return image.copy(), self.get_label(idx)
