from typing import List, Tuple
import random
import csv
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
import pandas as pd
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm

from .TabularAttributes import check_categorical_data, CAT_FEATURES, NUM_FEATURES, CAT_FEATURES_WITH_LABEL


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
      labels_path: str, img_size: int, live_loading: bool, missing_values: list = [], use_cache: bool = False, use_transformer: bool = False) -> None:
            
    # Imaging
    self.data_imaging = torch.load(data_path_imaging, map_location='cuda')
    self.transform = augmentation
    self.delete_segmentation = delete_segmentation
    self.augmentation_rate = augmentation_rate
    self.live_loading = live_loading
    self.use_cache = use_cache

    if self.delete_segmentation:
      for im in self.data_imaging:
        im[0,:,:] = 0

    self.default_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.Resize(size=(img_size,img_size)),
      transforms.ToTensor(),
      transforms.Lambda(lambda x : x.float())
    ])

    if self.use_cache:
      print('Caching images')
      self.cache_list = []
      self.cache_list_original = []
      for i in tqdm(range(len(self.data_imaging))):
        self.transforms_and_cache_images(i)

    # Tabular
    use_header = True if use_transformer else False
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.field_lengths = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular, missing_values, use_header)
    
    # Classifier
    self.labels = torch.load(labels_path)
  
  def read_and_parse_csv(self, path_tabular: str, missing_values: list = [], use_header: bool = False) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    if use_header:
      df = pd.read_csv(path_tabular)
      df.drop(missing_values, axis=0, inplace=True)
      cat_mask = check_categorical_data(df)
      self.cat_mask = cat_mask
      field_lengths_tensor = torch.tensor(self.field_lengths)
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
    print(f'Loaded tabular data from {path_tabular}')
    return data

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path)
    self.marginal_distributions = data_df.transpose().values.tolist()
    print('Generated marginal distributions')

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

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
    im = self.data_imaging[index]
    if self.live_loading:
      im = cv2.imread(im)
      im = im / 255
      im = im.astype("uint8")
    ims = [self.transform(im)]
    if random.random() < self.augmentation_rate:
      ims.append(self.transform(im))
    else:
      ims.append(self.default_transform(im))

    orig_im = self.default_transform(im)
    
    return ims, orig_im

  def transforms_and_cache_images(self, index: int) -> List[torch.Tensor]: 
    """
    Caches the augmented images for later use in a list
    """
    ims, orig_im = self.generate_imaging_views(index)
    # ims = [torch.tensor(im) for im in ims]
    self.cache_list.append(ims)
    self.cache_list_original.append(orig_im)
    return

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    if self.use_cache:
      imaging_views = self.cache_list[index]
      unaugmented_image = self.cache_list_original[index]
    else:
      imaging_views, unaugmented_image = self.generate_imaging_views(index)
    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]
    label = torch.tensor(self.labels[index], dtype=torch.long)
    return imaging_views, tabular_views, label, unaugmented_image

  def __len__(self) -> int:
    return len(self.data_tabular)