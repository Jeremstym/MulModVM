#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adapt tabular input to input tokens for transformer model

from typing import Literal, Tuple

import torch
from scipy.special import binom, factorial
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init


class PositionalEncoding(nn.Module):
    """Positional encoding layer."""

    def __init__(self, sequence_len: int, d_model: int):
        """Initializes layers parameters.

        Args:
            sequence_len: The number of tokens in the input sequence.
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        self.positional_encoding = Parameter(torch.empty(sequence_len, d_model))
        init.trunc_normal_(self.positional_encoding, std=0.2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that adds positional encoding to the input tensor.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, S, `d_model`), Tensor with added positional encoding.
        """
        return x + self.positional_encoding[None, ...]


class CLSToken(nn.Module):
    """[CLS]-token for BERT-like inference.

    When used as a module, the [CLS]-token is appended **to the end** of each item in the batch.

    Notes:
        - This is a port of the `CLSToken` class from v0.0.13 of the `rtdl` package. It mixes the original
          implementation with the simpler code of `_CLSEmbedding` from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L380-L446

    Examples:
        .. testcode::

            batch_size = 2
            n_tokens = 3
            d_token = 4
            cls_token = CLSToken(d_token, 'uniform')
            x = torch.randn(batch_size, n_tokens, d_token)
            x = cls_token(x)
            assert x.shape == (batch_size, n_tokens + 1, d_token)
            assert (x[:, -1, :] == cls_token.expand(len(x))).all()
    """

    def __init__(self, d_token: int) -> None:
        """Initializes class instance.

        Args:
            d_token: the size of token
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the weights using a uniform distribution."""
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying [CLS]-token to a tensor with the given leading dimensions.

        A possible use case is building a batch of [CLS]-tokens.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the underlying :code:`weight` parameter, so
            gradients will be propagated as expected.

        Args:
            leading_dimensions: the additional new dimensions

        Returns:
            tensor of the shape :code:`(*leading_dimensions, len(self.weight))`
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch (see `CLSToken`)."""
        return torch.cat([x, self.expand(len(x), 1)], dim=1)


class SequencePooling(nn.Module):
    """Sequence pooling layer."""

    def __init__(self, d_model: int):
        """Initializes layer submodules.

        Args:
            d_model: The number of features in the input (i.e. the dimensionality of the tokens).
        """
        super().__init__()
        # Initialize the learnable parameters of the sequential pooling
        self.attention_pool = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass that performs a (learnable) weighted averaging of the different tokens.

        Args:
            x: (N, S, `d_model`), Input tensor.

        Returns:
            (N, `d_model`), Output tensor.
        """
        attn_vector = F.softmax(self.attention_pool(x), dim=1)  # (N, S, 1)
        broadcast_attn_vector = attn_vector.transpose(2, 1)  # (N, S, 1) -> (N, 1, S)
        pooled_x = (broadcast_attn_vector @ x).squeeze(1)  # (N, 1, S) @ (N, S, E) -> (N, E)
        return pooled_x


class FTPredictionHead(nn.Module):
    """Prediction head architecture described in the Feature Tokenizer transformer (FT-Transformer) paper."""

    def __init__(self, in_features: int, out_features: int, is_classification: bool = True):
        """Initializes class instance.

        Args:
            in_features: Number of features in the input feature vector.
            out_features: Number of features to output.
        """
        super().__init__()
        self.output = nn.Sigmoid if is_classification else nn.Identity
        self.head = nn.Sequential(nn.LayerNorm(in_features), nn.ReLU(), nn.Linear(in_features, out_features), self.output())

    def forward(self, x: Tensor) -> Tensor:
        """Predicts unnormalized features from a feature vector input.

        Args:
            x: (N, `in_features`), Batch of feature vectors.

        Returns:
            - (N, `out_features`), Batch of output features.
        """
        if type(x) == tuple:
            x = x[0]

        return self.head(x)
