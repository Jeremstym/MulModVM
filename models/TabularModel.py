from typing import Dict
from collections import OrderedDict

import torch
import torch.nn as nn

from models.TabularEncoder import TabularEncoder
from models.TabularTransformer import TabularTransformer

class TabularModel(nn.Module):
  """
  Evaluation model for tabular trained with MLP backbone.
  """
  def __init__(self, args):
    super(TabularModel, self).__init__()

    self.encoder = TabularTransformer(args) if args.tabular_model == 'transformer' else TabularEncoder(args)
    if args.tabular_model == 'transformer' and args.use_xtab:
      self.load_pretrained_xtab(args)
    self.classifier = nn.Linear(args.tabular_embedding_dim, args.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.encoder(x)
    x = self.classifier(x)
    return x
    
  def load_pretrained_xtab(self, args) -> None:
    """
    Can load tabular encoder with pretrained weights from XTab foundation model
    """
    loaded_chkpt = torch.load(args.xtab_path, map_location=torch.device('cuda')) # load on GPU !
    self.encoder.load_state_dict(loaded_chkpt, strict=False) # no state_dict key needed as it is the whole state_dict
    learned_layer = [layer for layer in self.encoder_tabular.state_dict()]
    xtab_layer = [layer for layer in loaded_chkpt.keys()]
    intersection = set(learned_layer).intersection(set(xtab_layer))
    assert len(intersection) > 0, "No layers in common between learned model and XTab model"
    print(f"Loaded XTab model with layers: {intersection}")
    return
