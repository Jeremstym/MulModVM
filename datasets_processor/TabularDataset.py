from typing import Tuple
import csv
import pandas as pd

import torch
from torch.utils.data import Dataset
from .TabularAttributes import (
    check_categorical_data,
    CAT_FEATURES,
    NUM_FEATURES,
    CAT_FEATURES_WITH_LABEL,
)


class TabularDataset(Dataset):
    """ "
    Dataset for the evaluation of tabular data
    """

    def __init__(
        self,
        data_path: str,
        labels_path: str,
        eval_one_hot: bool = True,
        field_lengths_tabular: str = None,
        use_header: bool = True,
    ):
        super(TabularDataset, self).__init__()
        self.use_header = use_header
        self.labels = torch.load(labels_path)
        self.eval_one_hot = eval_one_hot
        self.field_lengths = torch.load(field_lengths_tabular)
        self.data = self.read_and_parse_csv(data_path)

        if self.eval_one_hot:
            for i in range(len(self.data)):
                self.data[i] = self.one_hot_encode(torch.tensor(self.data[i]))
        else:
            self.data = torch.as_tensor(self.data, dtype=torch.float)

    def get_input_size(self) -> int:
        """
        Returns the number of fields in the table.
        Used to set the input number of nodes in the MLP
        """
        if self.eval_one_hot:
            return int(sum(self.field_lengths))
        else:
            return len(self.data[0])

    def read_and_parse_csv(
        self,
        path_tabular: str,
        missing_values: list = [],
        use_header: bool = False,
        max_size: int = None,
    ) -> List[List[float]]:
        """
        Does what it says on the box.
        """
        #TODO: add an additional token when using label as a feature
        # if use_header and self._use_labels:
        #     FEATURES = NUM_FEATURES + CAT_FEATURES_WITH_LABEL
        if use_header:
            FEATURES = NUM_FEATURES + CAT_FEATURES
            df = pd.read_csv(path_tabular, names=FEATURES)
            df.drop(missing_values, axis=0, inplace=True)
            cat_mask = check_categorical_data(df)
            self.cat_mask = cat_mask
            field_lengths_tensor = torch.as_tensor(self.field_lengths_tabular)
            self.cat_card = field_lengths_tensor[cat_mask]
            data = df.values.tolist()
        else:
            with open(path_tabular, "r") as f:
                reader = csv.reader(f)
                data = []
                if max_size is not None:
                    for idx, r in enumerate(reader):
                        if idx in missing_values:
                            continue
                        r2 = [float(r1) for r1 in r]
                        data.append(r2)
                        if idx >= max_size:
                            break
                else:
                    for idx, r in enumerate(reader):
                        if idx in missing_values:
                            continue
                        r2 = [float(r1) for r1 in r]
                        data.append(r2)
        return data
        
    # def read_and_parse_csv(self, path: str):
    #     """
    #     Does what it says on the box
    #     """
    #     if self.use_header:
    #         df = pd.read_csv(path)
    #         cat_mask = check_categorical_data(df)
    #         self.cat_mask = cat_mask
    #         field_lengths_tensor = torch.tensor(self.field_lengths)
    #         self.cat_card = field_lengths_tensor[cat_mask]
    #         data = df.values.tolist()
    #     else:
    #         print("WARNING: dataframe has no headers for tokenization")
    #         with open(path, "r") as f:
    #             reader = csv.reader(f)
    #             data = []
    #             for r in reader:
    #                 r2 = [float(r1) for r1 in r]
    #                 data.append(r2)
    #     return data

    def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
        """
        One-hot encodes a subject's features
        """
        out = []
        for i in range(len(subject)):
            if self.field_lengths[i] == 1:
                out.append(subject[i].unsqueeze(0))
            else:
                out.append(
                    torch.nn.functional.one_hot(
                        torch.clamp(
                            subject[i], min=0, max=self.field_lengths[i] - 1
                        ).long(),
                        num_classes=int(self.field_lengths[i]),
                    )
                )
        return torch.cat(out)

    def get_cat_mask(self) -> torch.Tensor:
        """
        Returns the categorical mask
        """
        return torch.tensor(self.cat_mask)

    def get_cat_card(self) -> torch.Tensor:
        """
        Returns the categorical cardinalities
        """
        return torch.tensor(self.cat_card)

    def get_number_of_numerical_features(self) -> int:
        """
        Returns the number of numerical features
        """
        return len(NUM_FEATURES)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.labels[index]

    def __len__(self) -> int:
        return len(self.data)
