import torch
import torch.nn as nn

import hydra

from models.TokenLayers import PositionalEncoding, CLSToken
from datasets_processor.TabularAttributes import CAT_FEATURES, NUM_FEATURES

from typing import List, Tuple, Optional, Union, Dict
from torch import Tensor

class FusionCoreConcat(nn.Module):
    """
    Core module for fusing image and tabular data.
    """
    def __init__(self, args):
        super(FusionCore, self).__init__()
        self.image_dim = args.image_dim
        self.tabular_dim = args.tabular_dim
        self.fusion_dim = args.fusion_dim
        self.dropout = args.dropout

        self.image_linear = nn.Linear(self.image_dim, self.fusion_dim)
        self.tabular_linear = nn.Linear(self.tabular_dim, self.fusion_dim)
        self.fusion_linear = nn.Linear(self.fusion_dim * 2, self.fusion_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, image: Tensor, tabular: Tensor) -> Tensor:
        """
        Forward pass for the fusion core.
        """
        image = self.image_linear(image)
        tabular = self.tabular_linear(tabular)
        fusion = torch.cat((image, tabular), dim=1)
        fusion = self.relu(self.fusion_linear(fusion))
        fusion = self.dropout(fusion)
        return fusion

class FusionCoreCrossAtt(nn.Module):

    def __init__(self, args):
        super(FusionCoreCrossAtt, self).__init__()
        self.image_dim = args.img_size
        self.tabular_dim = args.tabular_embedding_dim
        self.fusion_dim = args.hidden_size
        assert self.tabular_dim == self.fusion_dim, "Tabular and fusion dimensions must be equal for cross-attention"
        assert self.image_dim % args.patch_size == 0, "Image dimensions must be divisible by the patch size"
        self.num_patches = (args.img_size // args.patch_size) ** 2
        self.num_tab_tokens = len(NUM_FEATURES) + len(CAT_FEATURES) + 1

        self.cls_token = CLSToken(d_token=self.fusion_dim)
        sequence_len = self.num_patches + self.num_tab_tokens
        self.positional_encoding_image = PositionalEncoding(self.num_patches, self.fusion_dim)
        self.positional_encoding_tabular = PositionalEncoding(self.num_tab_tokens, self.fusion_dim)       
        
        self.fusion = hydra.utils.instantiate(args.fusion_core)

    def forward(self, image: Tensor, tabular: Tensor) -> Tensor:
        """
        Forward pass for the fusion core.
        """
        # tabular = self.cls_token(tabular) | Already done in TabularTransformer
        tabular = self.positional_encoding_tabular(tabular) # is this necessary?
        image = self.positional_encoding_image(image)
        fusion = self.fusion(image, tabular)
        return fusion