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

        if args.checkpoint:
            loaded_chkpt = torch.load(args.checkpoint)
            original_args = loaded_chkpt['hyper_parameters']
            state_dict = loaded_chkpt['state_dict']
            self.input_size = original_args['input_size']
            
            if 'encoder_tabular.encoder.1.running_mean' in state_dict.keys():
                encoder_name = 'encoder_tabular.encoder.'
                self.encoder = self.build_encoder(original_args)
            elif 'encoder_projector_tabular.encoder.2.running_mean' in state_dict.keys():
                encoder_name = 'encoder_projector_tabular.encoder.'
                self.encoder = self.build_encoder_bn_old(original_args)
            else:
                encoder_name = 'encoder_projector_tabular.encoder.'
                self.encoder = self.build_encoder_no_bn(original_args)

            state_dict_encoder = {}
            for k in list(state_dict.keys()):
                if k.startswith(encoder_name):
                    state_dict_encoder[k[len(encoder_name):]] = state_dict[k]
            
            _ = self.encoder.load_state_dict(state_dict_encoder, strict=True)

            if args.finetune_strategy == 'frozen':
                for _, param in self.encoder.named_parameters():
                    param.requires_grad = False
                parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
                assert len(parameters)==0

        self.cls_token = CLSToken(d_token=args.d_token)
        sequence_len = len(NUM_FEATURES) + len(CAT_FEATURES) + 1 # +1 for CLS token
        self.positional_encoding = PositionalEncoding(sequence_len, args.d_token)
        self.TransformerEncoder = hydra.utils.instantiate(args.tabular_transformer)
        # self.head = nn.Linear(args.d_token, args.d_token)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cls_token(x)
        x = self.positional_encoding(x)
        x = self.TransformerEncoder(x)
        # x = self.head(x)
        return x[:, -1, :]