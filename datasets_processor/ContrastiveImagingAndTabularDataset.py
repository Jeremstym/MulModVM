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


class ContrastiveImagingAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of imaging and tabular data for contrastive learning.

  The first imaging view is always augmented. The second has {augmentation_rate} chance of being augmented.
  The first tabular view is never augmented. The second view is corrupted by replacing {corruption_rate} features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self, 
      data_path_imaging: str, delete_segmentation: bool, augmentation: transforms.Compose, augmentation_rate: float, 
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int, live_loading: bool, missing_values: list = [], 
      use_transformer: bool = False, use_labels: bool = False, use_embds: bool = False, use_lmdb=True) -> None:
            
    # Imaging
    self.transform = augmentation
    if use_embds:
      print('Using embeddings. IMPORTATION... Might take a while.')
    self.data_imaging = torch.load(data_path_imaging)
    if use_embds:
      print('Embeddings imported.')
    # if use_lmdb:
    #   self.env = lmdb.open("/home/stympopper/data/DVMdata/resized_DVM", map_size=int(1e12) , lock=False, readahead=False, meminit=False)
    #   with self.env.begin(write=True) as txn:
    #     for image_path in tqdm(self.data_imaging, desc='Creating LMDB', total=len(self.data_imaging)):
    #       im = cv2.imread(image_path)
    #       im = im / 255
    #       im = im.astype("uint8")
    #       im = self.transform(im)
    #       txn.put(image_path.encode(), pickle.dumps(im, protocol=DEFAULT_PROTOCOL))
    
    # raise Exception('LMDB created. Rerun script without this block.')
          

    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    # self.use_cache = use_cache
    self.use_labels = use_labels
    self.use_embds = use_embds

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(size=(img_size,img_size)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.float())
    ])

    # augmented_data = self.create_augmented_dataset(self.data_imaging, self.transform)
    # torch.save(augmented_data, '/home/stympopper/data/DVMdata/features/augmented_image_data.pt')
    # raise Exception('Augmented data saved to disk. Rerun script without this block.')

    # if self.use_cache:
    #   print('Caching images')
    #   self.cache_list = []
    #   self.cache_list_original = []
    #   for i in tqdm(range(len(self.data_imaging))):
    #     self.transforms_and_cache_images(i)

    # Tabular
    use_header = True if use_transformer else False
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular, missing_values, use_header)
    self.generate_marginal_distributions(data_path_tabular)
    
    # Classifier
    self.labels = torch.load(labels_path)
  
  def read_and_parse_csv(self, path_tabular: str, missing_values: list = [], use_header: bool = False) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    if use_header and self.use_labels:
      FEATURES = NUM_FEATURES + CAT_FEATURES_WITH_LABEL
    elif use_header:
      FEATURES = NUM_FEATURES + CAT_FEATURES
      df = pd.read_csv(path_tabular, names=FEATURES)
      df.drop(missing_values, axis=0, inplace=True)
      cat_mask = check_categorical_data(df)
      self.cat_mask = cat_mask
      field_lengths_tensor = torch.tensor(self.field_lengths_tabular)
      self.cat_card = field_lengths_tensor[cat_mask]
      data = df.values.tolist()
    else:
      with open(path_tabular,'r') as f:
        reader = csv.reader(f)
        data = []
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
      return len(self.data_tabular)

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

  def create_mask(self, subject: List[float]) -> List[bool]:
    """
    Creates a mask of features to be corrupted
    """
    mask = [False] * len(subject)
    random.seed(42)
    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      mask[i] = True
    return mask

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def generate_imaging_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects image. Also returns original image resized to required dimensions.
    The first is always augmented. The second has {augmentation_rate} chance to be augmented.
    """
    if not self.use_embds:
      im = self.data_imaging[index]
      if self.live_loading:
        im = cv2.imread(im)
        im = im / 255
        im = im.astype("uint8")
      ims = [self.transform(im)]
    else: 
      im = self.data_imaging[index]
      ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    
    return ims, orig_im

  # def create_augmented_dataset(self, dataset: Dataset, transform: Callable) -> Dataset:
  #   """
  #   Creates a new dataset with augmented images to save to disk
  #   """
  #   data_pipeline = []
  #   for i in tqdm(range(len(dataset)), desc='Augmenting data', total=len(dataset)):
  #     ims, orig_im = self.generate_imaging_views(i)
  #     data_pipeline.append(ims)
  #   return data_pipeline
    # with ThreadPool(8) as p:
    #   data_pipeline = list(tqdm(p.imap(self.generate_imaging_views, range(len(dataset))), total=len(dataset), desc='Augmenting data'))
    # augmented_data = []
    # for ims, orig_im in data_pipeline:
    #   augmented_data.append(ims)
    # return augmented_data

    # for i in tqdm(range(len(dataset)), desc='Augmenting data', total=len(dataset)):
    #   ims = self.generate_imaging_views(i)[0]
    #   augmented_data.append(ims)
    # return augmented_data


  # def transforms_and_cache_images(self, index: int) -> List[torch.Tensor]: 
  #   """
  #   Caches the augmented images for later use in a list
  #   """
  #   ims, orig_im = self.generate_imaging_views(index)
  #   # ims = [torch.tensor(im) for im in ims]
  #   self.cache_list.append(ims)
  #   self.cache_list_original.append(orig_im)
  #   return

  def get_cat_mask(self) -> torch.Tensor:
    """
    Returns the categorical mask
    """
    return torch.tensor(self.cat_mask)

  def get_cat_card(self) -> torch.Tensor:
    """
    Returns the categorical cardinalities
    """
    return torch.tensor(self.cat_card)

  def get_number_of_numerical_features(self) -> int:
    """
    Returns the number of numerical features
    """
    return len(NUM_FEATURES)

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    # if self.use_cache:
    #   imaging_views = self.cache_list[index]
    #   unaugmented_image = self.cache_list_original[index]
    # else:
    imaging_views, unaugmented_image = self.generate_imaging_views(index)
    if self.use_transformer:
      tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.create_mask(self.data_tabular[index]))]
    else:
      tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)

class CacheDataset(ContrastiveImagingAndTabularDataset):
  """
  Dataset with cache mechanism that can load data and cache deterministic transforms' result during training.

  By caching the results of non-random preprocessing transforms, it accelerates the training data pipeline.
  If the requested data is not in the cache, all transforms will run normally
  (see also :py:class:`monai.data.dataset.Dataset`).

  Users can set the cache rate or number of items to cache.
  It is recommended to experiment with different `cache_num` or `cache_rate` to identify the best training speed.

  The transforms which are supposed to be cached must implement the `monai.transforms.Transform`
  interface and should not be `Randomizable`. This dataset will cache the outcomes before the first
  `Randomizable` `Transform` within a `Compose` instance.
  So to improve the caching efficiency, please always put as many as possible non-random transforms
  before the randomized ones when composing the chain of transforms.
  If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
  for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

  For example, if the transform is a `Compose` of::

      transforms = Compose([
          LoadImaged(),
          EnsureChannelFirstd(),
          Spacingd(),
          Orientationd(),
          ScaleIntensityRanged(),
          RandCropByPosNegLabeld(),
          ToTensord()
      ])

  when `transforms` is used in a multi-epoch training pipeline, before the first training epoch,
  this dataset will cache the results up to ``ScaleIntensityRanged``, as
  all non-random transforms `LoadImaged`, `EnsureChannelFirstd`, `Spacingd`, `Orientationd`, `ScaleIntensityRanged`
  can be cached. During training, the dataset will load the cached results and run
  ``RandCropByPosNegLabeld`` and ``ToTensord``, as ``RandCropByPosNegLabeld`` is a randomized transform
  and the outcome not cached.

  During training call `set_data()` to update input data and recompute cache content, note that it requires
  `persistent_workers=False` in the PyTorch DataLoader.

  Note:
      `CacheDataset` executes non-random transforms and prepares cache content in the main process before
      the first epoch, then all the subprocesses of DataLoader will read the same cache content in the main process
      during training. it may take a long time to prepare cache content according to the size of expected cache data.
      So to debug or verify the program before real training, users can set `cache_rate=0.0` or `cache_num=0` to
      temporarily skip caching.

  Lazy Resampling:
      If you make use of the lazy resampling feature of `monai.transforms.Compose`, please refer to
      its documentation to familiarize yourself with the interaction between `CacheDataset` and
      lazy resampling.

  """

  def __init__(
      self,
      data_path_imaging: Sequence,
      delete_segmentation: bool,
      augmentation: transforms.Compose,
      augmentation_rate: float,
      data_path_tabular: str,
      corruption_rate: float,
      field_lengths_tabular: str,
      one_hot_tabular: bool,
      labels_path: str,
      img_size: int,
      live_loading: bool,
      missing_values: list = [],
      use_transformer: bool = True,
      use_labels: bool = False,
      transform: Sequence[Callable] | Callable | None = None,
      cache_num: int = sys.maxsize,
      cache_rate: float = 1.0,
      num_workers: int | None = 1,
      progress: bool = True,
      copy_cache: bool = True,
      as_contiguous: bool = False,
      hash_as_key: bool = False,
      # hash_func: Callable[..., bytes] = pickle_hashing,
      runtime_cache: bool | str | list | ListProxy = False,
      use_lmdb: bool = False,
  ) -> None:
      """
      Args:
          data: input data to load and transform to generate dataset for model.
          transform: transforms to execute operations on input data.
          cache_num: number of items to be cached. Default is `sys.maxsize`.
              will take the minimum of (cache_num, data_length x cache_rate, data_length).
          cache_rate: percentage of cached data in total, default is 1.0 (cache all).
              will take the minimum of (cache_num, data_length x cache_rate, data_length).
          num_workers: the number of worker threads if computing cache in the initialization.
              If num_workers is None then the number returned by os.cpu_count() is used.
              If a value less than 1 is specified, 1 will be used instead.
          progress: whether to display a progress bar.
          copy_cache: whether to `deepcopy` the cache content before applying the random transforms,
              default to `True`. if the random transforms don't modify the cached content
              (for example, randomly crop from the cached image and deepcopy the crop region)
              or if every cache item is only used once in a `multi-processing` environment,
              may set `copy=False` for better performance.
          as_contiguous: whether to convert the cached NumPy array or PyTorch tensor to be contiguous.
              it may help improve the performance of following logic.
          hash_as_key: whether to compute hash value of input data as the key to save cache,
              if key exists, avoid saving duplicated content. it can help save memory when
              the dataset has duplicated items or augmented dataset.
          hash_func: if `hash_as_key`, a callable to compute hash from data items to be cached.
              defaults to `monai.data.utils.pickle_hashing`.
          runtime_cache: mode of cache at the runtime. Default to `False` to prepare
              the cache content for the entire ``data`` during initialization, this potentially largely increase the
              time required between the constructor called and first mini-batch generated.
              Three options are provided to compute the cache on the fly after the dataset initialization:

              1. ``"threads"`` or ``True``: use a regular ``list`` to store the cache items.
              2. ``"processes"``: use a ListProxy to store the cache items, it can be shared among processes.
              3. A list-like object: a users-provided container to be used to store the cache items.

              For `thread-based` caching (typically for caching cuda tensors), option 1 is recommended.
              For single process workflows with multiprocessing data loading, option 2 is recommended.
              For multiprocessing workflows (typically for distributed training),
              where this class is initialized in subprocesses, option 3 is recommended,
              and the list-like object should be prepared in the main process and passed to all subprocesses.
              Not following these recommendations may lead to runtime errors or duplicated cache across processes.

      """
      # if not isinstance(transform, transforms.Compose):
      #     transform = transforms.Compose(transform)
      # super().__init__(data=data, transform=transform)
      super().__init__(
        data_path_imaging, delete_segmentation, augmentation, augmentation_rate, 
        data_path_tabular, corruption_rate, field_lengths_tabular, one_hot_tabular,
        labels_path, img_size, live_loading, missing_values, use_transformer, use_labels, False,
        use_lmdb=False
      )
      self.transform = augmentation
      self.delete_segmentation = delete_segmentation
      self.augmentation_rate = augmentation_rate
      self.live_loading = live_loading
      self.set_num = cache_num  # tracking the user-provided `cache_num` option
      self.set_rate = cache_rate  # tracking the user-provided `cache_rate` option
      self.progress = progress
      self.copy_cache = copy_cache
      self.as_contiguous = as_contiguous
      self.hash_as_key = hash_as_key
      # self.hash_func = hash_func
      self.num_workers = num_workers
      if self.num_workers is not None:
          self.num_workers = max(int(self.num_workers), 1)
      self.runtime_cache = runtime_cache
      self.cache_num = 0
      self._cache: list | ListProxy = []
      self._hash_keys: list = []
      self.data = torch.load(data_path_imaging)
      self.set_data(self.data)

      if self.delete_segmentation:
        for im in self.data_imaging:
          im[0,:,:] = 0

      self.default_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(img_size,img_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x : x.float())
      ])

      # Tabular
      use_header = True if use_transformer else False
      self.c = corruption_rate
      self.field_lengths_tabular = torch.load(field_lengths_tabular)
      self.one_hot_tabular = one_hot_tabular
      self.data_tabular = self.read_and_parse_csv(data_path_tabular, missing_values, use_header)
      self.generate_marginal_distributions(data_path_tabular)
      
      # Classifier
      self.labels = torch.load(labels_path)

      # Masking
      self.use_transformer = use_transformer

  def set_data(self, data: Sequence) -> None:
      """
      Set the input data and run deterministic transforms to generate cache content.

      Note: should call this func after an entire epoch and must set `persistent_workers=False`
      in PyTorch DataLoader, because it needs to create new worker processes based on new
      generated cache content.

      """
      self.data = data

      def _compute_cache_num(data_len: int):
          self.cache_num = min(int(self.set_num), int(data_len * self.set_rate), data_len)

      if self.hash_as_key:
          # only compute cache for the unique items of dataset, and record the last index for duplicated items
          mapping = {self.hash_func(v): i for i, v in enumerate(self.data)}
          _compute_cache_num(len(mapping))
          self._hash_keys = list(mapping)[: self.cache_num]
          indices = list(mapping.values())[: self.cache_num]
      else:
          _compute_cache_num(len(self.data))
          indices = list(range(self.cache_num))

      if self.runtime_cache in (False, None):  # prepare cache content immediately
          self._cache = self._fill_cache(indices)
          return
      if isinstance(self.runtime_cache, str) and "process" in self.runtime_cache:
          # this must be in the main process, not in dataloader's workers
          self._cache = Manager().list([None] * self.cache_num)
          return
      if (self.runtime_cache is True) or (isinstance(self.runtime_cache, str) and "thread" in self.runtime_cache):
          self._cache = [None] * self.cache_num
          return
      self._cache = self.runtime_cache  # type: ignore
      return

  def _fill_cache(self, indices=None) -> list:
      """
      Compute and fill the cache content from data source.

      Args:
          indices: target indices in the `self.data` source to compute cache.
              if None, use the first `cache_num` items.

      """
      if self.cache_num <= 0:
          return []
      if indices is None:
          indices = list(range(self.cache_num))
      if self.progress and not has_tqdm:
          warnings.warn("tqdm is not installed, will not show the caching progress bar.")
      with ThreadPool(self.num_workers) as p:
          if self.progress and has_tqdm:
              return list(tqdm(p.imap(self._load_cache_item, indices), total=len(indices), desc="Loading dataset"))
          return list(p.imap(self._load_cache_item, indices))

  def _load_cache_item(self, idx: int):
      """
      Args:
          idx: the index of the input data sequence.
      """
      # item = self.data[idx]
      item_augmented, item_not_augmented = self.generate_imaging_views(idx)

      # first_random = self.transform.get_index_of_first(
      #     lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
      # )
      # item = self.transform(item)

      if self.as_contiguous:
          item = convert_to_contiguous(item, memory_format=torch.contiguous_format)
      return item_augmented, item_not_augmented

  def _transform(self, index: int):
      cache_index = None
      if self.hash_as_key:
          key = self.hash_func(self.data[index])
          if key in self._hash_keys:
              # if existing in cache, try to get the index in cache
              cache_index = self._hash_keys.index(key)
      elif index % len(self) < self.cache_num:  # support negative index
          cache_index = index

      if cache_index is None:
          # no cache for this index, execute all the transforms directly
          return super().generate_imaging_views(index)

      if self._cache is None:
          raise RuntimeError("cache buffer is not initialized, please call `set_data()` first.")
      imaging_views, unaugmented_image = self._cache[cache_index]
      # runtime cache computation
      if imaging_views is None:
          imaging_views, unaugmented_image = self._cache[cache_index] = self._load_cache_item(cache_index)

      # load data from cache and execute from the first random transform
      if not isinstance(self.transform, transforms.Compose):
          raise ValueError("transform must be an instance of monai.transforms.Compose.")

      # first_random = self.transform.get_index_of_first(
      #     lambda t: isinstance(t, RandomizableTrait) or not isinstance(t, Transform)
      # )
      # if first_random is not None:
      #     data = deepcopy(data) if self.copy_cache is True else data
      #     data = self.transform(data)

      return imaging_views, unaugmented_image
  
  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
      # if self.use_cache:
      #   imaging_views = self.cache_list[index]
      #   unaugmented_image = self.cache_list_original[index]
      # else:
      imaging_views, unaugmented_image = self._transform(index)
      tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
      if self.one_hot_tabular:
        tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
      label = torch.tensor(self.labels[index], dtype=torch.long)
      return imaging_views, tabular_views, label, unaugmented_image