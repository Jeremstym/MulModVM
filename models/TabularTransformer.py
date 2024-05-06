#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Use FT-Transformer to encoder tabular data after tokenization

from typing import Dict
from collections import OrderedDict
import hydra

import torch 
import torch.nn as nn

from models.TokenLayers import PositionalEncoding, CLSToken
from datasets_processor.TabularAttributes import CAT_FEATURES, NUM_FEATURES

class TabularTransformer(nn.Module):
    """
    Evaluation model for tabular trained with FT-Transformer backbone.
    """
    def __init__(self, args):
        super(TabularTransformer, self).__init__()

        self.cls_token = CLSToken(d_token=args.embedding_dim)
        sequence_len = len(NUM_FEATURES) + len(CAT_FEATURES)
        self.positional_encoding = PositionalEncoding(sequence_len, args.embedding_dim)
        self.TransformerEncoder = hydra.utils.instantiate(args.tabular_transformer)
        self.head = nn.Linear(args.embedding_dim, args.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cls_token(x)
        x = self.positional_encoding(x)
        x = self.TransformerEncoder(x)
        x = self.head(x)
        return x