#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

CAT_FEATURES_WITH_LABEL = [
  'Color',
  'Bodytype',
  'Gearbox',
  'Fuel_type',
  'Genmodel_ID'
  ]

CAT_FEATURES = [
  'Color',
  'Bodytype',
  'Gearbox',
  'Fuel_type',
  ]

NUM_FEATURES = [
  'Wheelbase',
  'Height',
  'Width',
  'Length',
  'Adv_year',
  'Adv_month',
  'Reg_year',
  'Runned_Miles',
  'Price',
  'Seat_num',
  'Door_num',
  'Entry_price', 
  'Engine_size',]

FEATURES =  NUM_FEATURES + CAT_FEATURES

def check_categorical_data(df: pd.DataFrame):
  """
  Checks if the data is categorical or not
  """
  categorizing = [c in CAT_FEATURES for c in FEATURES]
  typing = [df[c].dtype == np.dtype('int64') for c in df.iloc[0]]
  assert all([c == t for c,t in zip(categorizing, typing)]), "Mismatch between categorical and numerical data"
  return categorizing