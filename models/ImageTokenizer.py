import torch
import torch.nn as nn
from torch import Tensor
import collections

from typing import List, Tuple, Optional, Union, Dict


class ViTTokenizer(nn.Module):
    """
    Tokenizer the image into tokens, following ViT (Vision Transformer) approach
    https://github.com/huggingface/transformers

    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, args):
        super().__init__()
        image_size, patch_size = args.img_size, args.patch_size
        num_channels, hidden_size = args.num_channels, args.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # In our framework, we require the image to be square
        assert image_size[0] == image_size[1], "Image size must be square"
        assert patch_size[0] == patch_size[1], "Patch size must be square"
        # In our framework, we require the image dimensions to be divisible by the patch size (no interpolation)
        assert image_size[0] % patch_size[0] == 0, "Image dimensions must be divisible by the patch size"
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        """ Generate sequence of image embeddings

        Args:
            pixel_values (torch.Tensor): image tensor of shape `(batch_size, num_channels, height, width)`.
            interpolate_pos_encoding (bool, optional): Interpolate if ???. Defaults to False.

        Returns:
            torch.Tensor: Image sequence embeddings of shape `(batch_size, num_patches, hidden_size)`.
        """
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2) # (batch_size, num_patches, hidden_size)
        return embeddings
