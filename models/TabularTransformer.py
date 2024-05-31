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

        # if args.checkpoint:
        #     loaded_chkpt = torch.load(args.checkpoint)
        #     original_args = loaded_chkpt['hyper_parameters']
        #     state_dict = loaded_chkpt['state_dict']
        #     self.input_size = original_args['input_size']
            
        #     if 'encoder_tabular.encoder.1.running_mean' in state_dict.keys():
        #         encoder_name = 'encoder_tabular.encoder.'
        #         self.encoder = self.build_encoder(original_args)
        #     elif 'encoder_projector_tabular.encoder.2.running_mean' in state_dict.keys():
        #         encoder_name = 'encoder_projector_tabular.encoder.'
        #         self.encoder = self.build_encoder_bn_old(original_args)
        #     else:
        #         encoder_name = 'encoder_projector_tabular.encoder.'
        #         self.encoder = self.build_encoder_no_bn(original_args)

        #     state_dict_encoder = {}
        #     for k in list(state_dict.keys()):
        #         if k.startswith(encoder_name):
        #             state_dict_encoder[k[len(encoder_name):]] = state_dict[k]
            
        #     _ = self.encoder.load_state_dict(state_dict_encoder, strict=True)

        #     if args.finetune_strategy == 'frozen':
        #         for _, param in self.encoder.named_parameters():
        #             param.requires_grad = False
        #         parameters = list(filter(lambda p: p.requires_grad, self.encoder.parameters()))
        #         assert len(parameters)==0

        self.cls_token = CLSToken(d_token=args.tabular_embedding_dim)
        sequence_len = len(NUM_FEATURES) + len(CAT_FEATURES) + 1 # +1 for CLS token
        self.positional_encoding = PositionalEncoding(sequence_len, args.tabular_embedding_dim)
        self.TransformerEncoder = hydra.utils.instantiate(args.tabular_transformer)
        # self.head = nn.Linear(args.d_token, args.d_token)

    def mask_tokens(self, x: torch.Tensor, mask_token: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Replaces tokens in a batch of sequences with a predefined `mask_token`.

        References:
            - Adapted from the random masking implementation from the paper that introduced Mask Token Replacement (MTR):
            https://github.com/somaonishi/MTR/blob/33b87b37a63d120aff24c041da711fd8b714c00e/model/mask_token.py#L52-L68

        Args:
            x: (N, S, E) Batch of sequences of tokens.
            mask_token: (E) or (S, E) Mask to replace masked tokens with. If a single token of dimension (E), then the mask
                will be used to replace any tokens in the sequence. Otherwise, each token in the sequence has to have its
                own MASK token to be replaced with.
            mask: (N, S) Boolean mask of tokens in each sequence, with (True) representing tokens to replace.

        Returns:
            (N, S, E) Input tokens, where the requested tokens have been replaced by the mask token.
        """
        n, s, d = x.shape

        broadcast_mask = mask.unsqueeze(-1).to(device=x.device, dtype=torch.float)
        broadcast_mask = broadcast_mask.repeat(1, 1, d)

        if mask_token.ndim == 1:
            mask_tokens = mask_token[None, None, ...].repeat(n, s, 1)
        elif mask_token.ndim == 2:
            mask_tokens = mask_token[None, ...].repeat(n, 1, 1)
        else:
            raise ValueError(
                "The `mask_token` parameter passed to `random_masking` should be of dimensions (E) or (S, E), where E is "
                "the embedding size and S is the sequence length."
            )

        x_masked = x * (1 - broadcast_mask) + mask_tokens * broadcast_mask
        return x_masked

    def masking_token(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        print(f'x size is: {x.size()}')
        raise Exception('stop')
        mask_token = torch.zeros(1, x.size(1), x.size(2)).to(x.device)
        x = self.mask_tokens(x, mask_token, mask)
        return x

    def forward(self, x: torch.Tensor, mask_corruption: torch.Tensor = None) -> torch.Tensor:
        if mask_corruption is not None:
            x = self.masking_token(x, mask_corruption)
        x = self.cls_token(x)
        x = self.positional_encoding(x)
        x = self.TransformerEncoder(x)
        # x = self.head(x)
        return x[:, -1, :]