from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from models.TabularTokenizer import TabularTokenizer
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel

class Fusion(pl.LightningModule):
    def __init__(self, hparams, dataset=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.datatype == 'multimodal' or self.hparams.datatype == 'imaging_and_tabular':
            self.model = MultimodalModel(self.hparams)
        else:
            raise ValueError('The only way to use the Fusion model is with multimodal or imaging_and_tabular data')

        task = 'binary' if self.hparams.num_classes == 2 else 'multiclass'

        self.acc_train = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_val = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)
        self.acc_test = torchmetrics.Accuracy(task=task, num_classes=self.hparams.num_classes)

        self.auc_train = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_val = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)
        self.auc_test = torchmetrics.AUROC(task=task, num_classes=self.hparams.num_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_score = 0

        print(self.model)

    def forward_tabular(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a prediction from a data point
        """
        x_num = x[:, ~self.cat_mask]
        x_cat = x[:, self.cat_mask].type(torch.int64)
        x = self.model.tokenizer_tabular(x_num=x_num, x_cat=x_cat)
        y_hat = self.model.encoder_tabular(x)
        return y_hat

    def forward_imaging(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates a prediction from a data point
        """
        y_hat = self.model.encoder_imaging(x)
        return y_hat

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Generates a prediction from a data point
        """
        x_tab, x_img = x
        y_hat_tab = self.forward_tabular(x_tab)
        y_hat_img = self.forward_imaging(x_img)
        y_hat = self.model.classifier(y_hat_tab, y_hat_img)
        return y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
        """
        Trains model
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Validates model
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Tests model
        """
        x, y = batch
        y_hat = self.forward(x)
        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """
        Test epoch end
        """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()
        self.log('test_acc', test_acc)
        self.log('test_auc', test_auc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer