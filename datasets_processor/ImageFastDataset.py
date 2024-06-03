from typing import Tuple
import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm

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
        labels: List[int] = None,
        max_size: int = None,
        **super_kwargs,
    ):
        self.data = data_path
        self._name = name
        self.labels = labels

        if os.path.isdir(self._path):
            self._type = "dir"
            self._all_fnames = {
                os.path.relpath(os.path.join(root, fname), start=self._path)
                for root, _dirs, files in os.walk(self._path)
                for fname in files
            }
        else:
            raise ValueError(f"Path {self._path} is not a directory")

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

    # @property
    # def label_shape(self):
    #     if self._label_shape is None:
    #         raw_labels = self._get_raw_labels()
    #         if raw_labels.dtype == np.int64:
    #             self._label_shape = [int(np.max(raw_labels)) + 1]
    #         else:
    #             self._label_shape = raw_labels.shape[1:]
    #     return list(self._label_shape)

    # @property
    # def label_dim(self):
    #     assert len(self.label_shape) == 1
    #     return self.label_shape[0]

    # @property
    # def has_labels(self):
    #     return any(x != 0 for x in self.label_shape)

    # @property
    # def has_onehot_labels(self):
    #     return self._get_raw_labels().dtype == np.int64

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(self._path, fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    # def close(self):
    #     try:
    #         if self._zipfile is not None:
    #             self._zipfile.close()
    #     finally:
    #         self._zipfile = None

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
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def get_image_from_idx(self, idx):
        return self._load_raw_image(self._raw_idx[idx])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        return image.copy()
