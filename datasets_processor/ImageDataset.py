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


# class CacheDataset(ImageDataset):
#     """
#     Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

#     By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
#     If the requested data is not in the cache, all transforms will run normally
#     (see also :py:class:`monai.data.dataset.Dataset`).

#     Users can set the cache rate or number of items to cache.
#     It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

#     The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
#     interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
#     `Randomizable` `Transform` within a `Compose` instance.
#     So to improve the caching efficiency, please always put as many as possible non-random transforms
#     before the randomized ones when composing the chain of transforms.
#     If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
#     for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

#     For example, if the transform is a `Compose` of::

#         transforms = Compose([
#             LoadImaged(),
#             EnsureChannelFirstd(),
#             Spacingd(),
#             Orientationd(),
#             ScaleIntensityRanged(),
#             RandCropByPosNegLabeld(),
#             ToTensord()
#         ])

#     when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
#     this dataset will cache the results up to ``ScaleIntensityRanged``, as
#     all non-random transforms `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
#     can be cached. During training, the dataset will load the cached results and run
#     ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
#     and the outcome not cached.

#     During training call `set_data()` to update input data and recompute cache content, note that it requires
#     `persistent_workers=False` in the PyTorch DataLoader.

#     Note:
#         `CacheDataset` executes non-random transforms and prepares cache content in the main process before
#         the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
#         during training. it may take a long time to prepare cache content according to the size of expected cache data.
#         So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
#         temporarily skip caching.

#     Lazy Resampling:
#         If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
#         its documentation to familiarize yourself with the interaction between `CacheDataset` and
#         lazy resampling.

#     """

#     def __init__(
#         self,
#         data: Sequence,
#         transform: Sequence[Callable] | Callable | None = None,
#         cache_num: int = sys.maxsize,
#         cache_rate: float = 1.0,
#         num_workers: int | None = 1,
#         progress: bool = True,
#         copy_cache: bool = True,
#         as_contiguous: bool = True,
#         hash_as_key: bool = False,
#         # hash_func: Callable[..., bytes] = pickle_hashing,
#         runtime_cache: bool | str | list | ListProxy = False,
#     ) -> None:
#         """
#         Args:
#             data: input data to load and transform to generate dataset for model.
#             transform: transforms to execute operations on input data.
#             cache_num: number of items to be cached. Default is `sys.maxsize`.
#                 will take the minimum of (cache_num, data_length x cache_rate, data_length).
#             cache_rate: percentage of cached data in total, default is 1.0 (cache all).
#                 will take the minimum of (cache_num, data_length x cache_rate, data_length).
#             num_workers: the number of worker threads if computing cache in the initialization.
#                 If num_workers is None then the number returned by os.cpu_count() is used.
#                 If a value less than 1 is specified, 1 will be used instead.
#             progress: whether to display a progress bar.
#             copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
#                 default to `True`. if the random transforms don't modify the cached content
#                 (for example, randomly crop from the cached image and deepcopy the crop region)
#                 or if every cache item is only used once in a `multi-processing` environment,
#                 may set `copy=False` for better performance.
#             as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
#                 it may help improve the performance of following logic.
#             hash_as_key: whether to compute hash value of input data as the key to save cache,
#                 if key exists, avoid saving duplicated content. it can help save memory when
#                 the dataset has duplicated items or augmented dataset.
#             hash_func: if `hash_as_key`, a callable to compute hash from data items to be cached.
#                 defaults to `monai.data.utils.pickle_hashing`.
#             runtime_cache: mode of cache at the runtime. Default to `False` to prepare
#                 the cache content for the entire ``data`` during initialization, this potentially largely increase the
#                 time required between the constructor called and first mini-batch generated.
#                 Three options are provided to compute the cache on the fly after the dataset initialization:

#                 1. ``"threads"`` or ``True``: use a regular ``list`` to store the cache items.
#                 2. ``"processes"``: use a ListProxy to store the cache items, it can be shared among processes.
#                 3. A list-like object: a users-provided container to be used to store the cache items.

#                 For `thread-based` caching (typically for caching cuda tensors), option 1 is recommended.
#                 For single process workflows with multiprocessing data loading, option 2 is recommended.
#                 For multiprocessing workflows (typically for distributed training),
#                 where this class is initialized in subprocesses, option 3 is recommended,
#                 and the list-like object should be prepared in the main process and passed to all subprocesses.
#                 Not following these recommendations may lead to runtime errors or duplicated cache across processes.

#         """
#         if not isinstance(transform, Compose):
#             transform = Compose(transform)
#         super().__init__(data=data, transform=transform)
#         self.set_num = cache_num  # tracking the user-provided `cache_num` option
#         self.set_rate = cache_rate  # tracking the user-provided `cache_rate` option
#         self.progress = progress
#         self.copy_cache = copy_cache
#         self.as_contiguous = as_contiguous
#         self.hash_as_key = hash_as_key
#         # self.hash_func = hash_func
#         self.num_workers = num_workers
#         if self.num_workers is not None:
#             self.num_workers = max(int(self.num_workers), 1)
#         self.runtime_cache = runtime_cache
#         self.cache_num = 0
#         self._cache: list | ListProxy = []
#         self._hash_keys: list = []
#         self.set_data(data)

#     def set_data(self, data: Sequence) -> None:
#         """
#         Set the input data and run deterministic transforms to generate cache content.

#         Note: should call this func after an entire epoch and must set `persistent_workers=False`
#         in PyTorch DataLoader, because it needs to create new worker processes based on new
#         generated cache content.

#         """
#         self.data = data

#         def _compute_cache_num(data_len: int):
#             self.cache_num = min(int(self.set_num), int(data_len * self.set_rate), data_len)

#         if self.hash_as_key:
#             # only compute cache for the unique items of dataset, and record the last index for duplicated items
#             mapping = {self.hash_func(v): i for i, v in enumerate(self.data)}
#             _compute_cache_num(len(mapping))
#             self._hash_keys = list(mapping)[: self.cache_num]
#             indices = list(mapping.values())[: self.cache_num]
#         else:
#             _compute_cache_num(len(self.data))
#             indices = list(range(self.cache_num))

#         if self.runtime_cache in (False, None):  # prepare cache content immediately
#             self._cache = self._fill_cache(indices)
#             return
#         if isinstance(self.runtime_cache, str) and "process" in self.runtime_cache:
#             # this must be in the main process, not in dataloader's workers
#             self._cache = Manager().list([None] * self.cache_num)
#             return
#         if (self.runtime_cache is True) or (isinstance(self.runtime_cache, str) and "thread" in self.runtime_cache):
#             self._cache = [None] * self.cache_num
#             return
#         self._cache = self.runtime_cache  # type: ignore
#         return

#     def _fill_cache(self, indices=None) -> list:
#         """
#         Compute and fill the cache content from data source.

#         Args:
#             indices: target indices in the `self.data` source to compute cache.
#                 if None, use the first `cache_num` items.

#         """
#         if self.cache_num <= 0:
#             return []
#         if indices is None:
#             indices = list(range(self.cache_num))
#         if self.progress and not has_tqdm:
#             warnings.warn("tqdm is not installed, will not show the caching progress bar.")
#         with ThreadPool(self.num_workers) as p:
#             if self.progress and has_tqdm:
#                 return list(tqdm(p.imap(self._load_cache_item, indices), total=len(indices), desc="Loading dataset"))
#             return list(p.imap(self._load_cache_item, indices))

#     def _load_cache_item(self, idx: int):
#         """
#         Args:
#             idx: the index of the input data sequence.
#         """
#         item = self.data[idx]

#         first_random = self.transform.get_index_of_first(
#             lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
#         )
#         item = self.transform(item, end=first_random, threading=True)

#         if self.as_contiguous:
#             item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
#         return item

#     def _transform(self, index: int):
#         cache_index = None
#         if self.hash_as_key:
#             key = self.hash_func(self.data[index])
#             if key in self._hash_keys:
#                 # if existing in cache, try to get the index in cache
#                 cache_index = self._hash_keys.index(key)
#         elif index % len(self) < self.cache_num:  # support negative index
#             cache_index = index

#         if cache_index is None:
#             # no cache for this index, execute all the transforms directly
#             return super()._transform(index)

#         if self._cache is None:
#             raise RuntimeError("cache buffer is not initialized, please call `set_data()` first.")
#         data = self._cache[cache_index]
#         # runtime cache computation
#         if data is None:
#             data = self._cache[cache_index] = self._load_cache_item(cache_index)

#         # load data from cache and execute from the first random transform
#         if not isinstance(self.transform, Compose):
#             raise ValueError("transform must be an instance of monai.transforms.Compose.")

#         first_random = self.transform.get_index_of_first(
#             lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
#         )
#         if first_random is not None:
#             data = deepcopy(data) if self.copy_cache is True else data
#             data = self.transform(data, start=first_random)

#         return data
