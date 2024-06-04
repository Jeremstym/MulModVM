import torch
import torch.nn as nn
from collections import OrderedDict

import hydra
from omegaconf import DictConfig

from models.TabularModel import TabularModel
from models.TabularTransformer import TabularTransformer
from models.TabularTokenizer import TabularTokenizer
from models.ImagingModel import ImagingModel

class MultimodalFusionModel(nn.Module):
    """
    Evaluation model for imaging and tabular data.
    """
    def __init__(
        self,
        args,
        cat_cardinalities: list,
        n_num_features: int,
        cat_mask: list,
        ) -> None:
        super().__init__()
    
        self.imaging_model = ImagingModel(args)
        self.tabular_encoder = TabularTransformer(args)
        # in_dim = 4096
        tab_dim = args.tabular_transformer.d_token
        self.tokenizer = hydra.utils.instantiate(args.tabular_tokenizer, cat_cardinalities=cat_cardinalities, n_num_features=n_num_features)
        self.tab_head = nn.Linear(tab_dim, args.projection_dim)
        self.im_head = nn.Linear(args.embedding_dim, args.projection_dim)
        self.head = nn.Linear(args.projection_dim*2, args.num_classes)
        self.cat_mask = cat_mask
    

    def tokenizer_tabular(self, x: torch.Tensor) -> torch.Tensor:
        x_num = x[:, ~self.cat_mask]
        x_cat = x[:, self.cat_mask].type(torch.int64)
        x = self.tokenizer(x_num=x_num, x_cat=x_cat)
        return x

    def encoder_tabular(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tokenizer_tabular(x)
        x = self.tabular_encoder(x)
        return x

    def encoder_imaging(self, x: torch.Tensor) -> torch.Tensor:
        if self.imaging_model.bolt_encoder:
            x = self.imaging_model.encoder(x)[0]
        else:
            x = self.imaging_model.encoder(x).squeeze()
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_im = self.encoder_imaging(x[0]) # only keep the encoder output
        x_proj_im = self.im_head(x_im)
        x_tab = self.encoder_tabular(x[1]).squeeze()
        x_proj_tab = self.tab_head(x_tab)
        x = torch.cat([x_proj_im, x_proj_tab], dim=1)
        x = self.head(x)
        return x