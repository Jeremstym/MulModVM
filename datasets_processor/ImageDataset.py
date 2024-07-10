from typing import Tuple
import random

import torch
from torch.utils.data import Dataset, Subset
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm
has_tqdm = True

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations
from .TabularAttributes import check_categorical_data, CAT_FEATURES, NUM_FEATURES, CAT_FEATURES_WITH_LABEL

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

# from monai.data.meta_tensor import MetaTensor
# from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
# from monai.transforms import (
#     Compose,
#     Randomizable,
#     RandomizableTrait,
#     Transform,
#     apply_transform,
#     convert_to_contiguous,
#     reset_ops_id,
# )
# from monai.utils import MAX_SEED, convert_to_tensor, get_seed, look_up_option, min_version, optional_import
# from monai.utils.misc import first

class ImageDataset(Dataset):
  """
  Dataset for the evaluation of images
  """
  def __init__(self, data: str, labels: str, delete_segmentation: bool, eval_train_augment_rate: float, img_size: int, target: str, train: bool, live_loading: bool, task: str) -> None:
    super(ImageDataset, self).__init__()
    self.train = train
    self.eval_train_augment_rate = eval_train_augment_rate
    self.live_loading = live_loading
    self.task = task

    self.data = torch.load(data)
    self.labels = torch.load(labels)

    if delete_segmentation:
      for im in self.data:
        im[0,:,:] = 0

    self.transform_train = grab_hard_eval_image_augmentations(img_size, target)
    self.transform_val = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(size=(img_size,img_size)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.float())
    ])

  def __getitem__(self, indx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns an image for evaluation purposes.
    If training, has {eval_train_augment_rate} chance of being augmented.
    If val, never augmented.
    """
    im = self.data[indx]
    if self.live_loading:
      im = cv2.imread(im)
      im = im / 255
      im = im.astype('uint8')

    if self.train and (random.random() <= self.eval_train_augment_rate):
      im = self.transform_train(im)
    else:
      im = self.transform_val(im)
    
    label = self.labels[indx]
    return (im), label

  def __len__(self) -> int:
    return len(self.labels)
