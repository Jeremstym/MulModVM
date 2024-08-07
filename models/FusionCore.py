import torch
import torch.nn as nn

import hydra

from models.TokenLayers import PositionalEncoding, CLSToken
from datasets_processor.TabularAttributes import CAT_FEATURES, NUM_FEATURES, NUM_FEATURS_NON_PHYSICAL

from typing import List, Tuple, Optional, Union, Dict
from torch import Tensor

class FusionCoreConcat(nn.Module):
    """
    Core module for fusing image and tabular data.
    """
    def __init__(self, args):
        super(FusionCoreConcat, self).__init__()

    def forward(self, image: Tensor, tabular: Tensor) -> Tensor:
        """
        Forward pass for the fusion core.
        """
        if tabular.ndim == 3:
            tabular = tabular[:, -1, :]
        fusion = torch.cat([image, tabular], dim=1)
        return fusion

class FusionCoreCrossAtt(nn.Module):

    def __init__(self, args):
        super(FusionCoreCrossAtt, self).__init__()
        self.image_dim = args.img_size
        self.tabular_dim = args.tabular_embedding_dim
        self.fusion_dim = args.hidden_size
        assert self.tabular_dim == self.fusion_dim, "Tabular and fusion dimensions must be equal for cross-attention"
        assert self.image_dim % args.patch_size == 0, "Image dimensions must be divisible by the patch size"
        if args.model == 'resnet18':
            self.num_patches = 36
        else:
            self.num_patches = (args.img_size // args.patch_size) ** 2
        if args.use_physical:
            self.num_tab_tokens = len(NUM_FEATURES) + len(CAT_FEATURES) + 1
        else:
            self.num_tab_tokens = len(NUM_FEATURS_NON_PHYSICAL) + len(CAT_FEATURES) + 1

        self.cls_token = CLSToken(d_token=self.tabular_dim)
        sequence_len = self.num_patches + self.num_tab_tokens
        self.positional_encoding_image = PositionalEncoding(self.num_patches, self.fusion_dim)
        self.positional_encoding_tabular = PositionalEncoding(self.num_tab_tokens, self.fusion_dim)       
        
        self.fusion = hydra.utils.instantiate(args.fusion_core)

    def forward(self, image: Tensor, tabular: Tensor) -> Tensor:
        """
        Forward pass for the fusion core.
        """
        tabular = self.cls_token(tabular) # Already done in TabularTransformer?
        tabular = self.positional_encoding_tabular(tabular) # is this necessary?
        image = self.positional_encoding_image(image)
        fusion = self.fusion(tabular, image)
        return fusion[:, self.num_tab_tokens - 1, :]