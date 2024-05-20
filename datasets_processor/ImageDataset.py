from typing import Tuple
import random

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from torchvision.io import read_image
import cv2
from tqdm import tqdm

from utils.utils import grab_hard_eval_image_augmentations, grab_soft_eval_image_augmentations, grab_image_augmentations
from .TabularAttributes import check_categorical_data, CAT_FEATURES, NUM_FEATURES, CAT_FEATURES_WITH_LABEL


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
