from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalFusionModel import MultimodalFusionModel


class Fusion(pl.LightningModule):
    def __init__(self, hparams, dataset=None):
        super().__init__()
        self.save_hyperparameters(hparams)

        assert dataset is not None, "Dataset must be provided for transformer models"
        cat_mask = dataset.get_cat_mask()
        self.cat_mask = cat_mask
        num_cont = dataset.get_number_of_numerical_features()
        cat_card = dataset.get_cat_card()

        if (
            self.hparams.datatype == "multimodal"
            or self.hparams.datatype == "imaging_and_tabular"
        ):
            self.model = MultimodalFusionModel(
                self.hparams,
                cat_cardinalities=cat_card.tolist(),
                n_num_features=num_cont,
                cat_mask=cat_mask,
            )
        else:
            raise ValueError(
                "The only way to use the Fusion model is with multimodal or imaging_and_tabular data"
            )

        task = "binary" if self.hparams.num_classes == 2 else "multiclass"

        self.acc_train = torchmetrics.Accuracy(
            task=task, num_classes=self.hparams.num_classes
        )
        self.acc_val = torchmetrics.Accuracy(
            task=task, num_classes=self.hparams.num_classes
        )
        self.acc_test = torchmetrics.Accuracy(
            task=task, num_classes=self.hparams.num_classes
        )

        self.auc_train = torchmetrics.AUROC(
            task=task, num_classes=self.hparams.num_classes
        )
        self.auc_val = torchmetrics.AUROC(
            task=task, num_classes=self.hparams.num_classes
        )
        self.auc_test = torchmetrics.AUROC(
            task=task, num_classes=self.hparams.num_classes
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.best_val_score = 0

        print(self.model)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Generates a prediction from a data point
        """
        y_hat = self.model(x)
        return y_hat

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], _
    ) -> torch.Tensor:
        """
        Trains model
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_train(y_hat, y)
        self.auc_train(y_hat, y)

        self.log("fusion_train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def training_epoch_end(self, _) -> None:
        """
        Training epoch end
        """
        self.log(
            "fusion_train_acc",
            self.acc_train,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.acc_train,
        )

        self.log(
            "fusion_train_auc",
            self.auc_train,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.auc_train,
        )

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Validates model
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_val(y_hat, y)
        self.auc_val(y_hat, y)

        self.log("fusion_val_loss", loss)

    def validation_epoch_end(self, _) -> None:
        """
        Validation epoch end
        """
        if self.trainer.sanity_checking:
            return

        epoch_val_acc = self.acc_val.compute()
        epoch_val_auc = self.auc_val.compute()

        self.log(
            "fusion_val_acc",
            epoch_val_acc,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.acc_val,
        )

        self.log(
            "fusion_val_auc",
            epoch_val_auc,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.auc_val,
        )

        self.best_val_score = max(self.best_val_score, epoch_val_acc)

        self.acc_test.reset()
        self.auc_test.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
        """
        Tests model
        """
        x, y = batch
        y_hat = self.forward(x)

        y_hat = torch.softmax(y_hat.detach(), dim=1)
        if self.hparams.num_classes == 2:
            y_hat = y_hat[:, 1]

        self.acc_test(y_hat, y)
        self.auc_test(y_hat, y)

    def test_epoch_end(self, _) -> None:
        """
        Test epoch end
        """
        test_acc = self.acc_test.compute()
        test_auc = self.auc_test.compute()

        self.log("test.acc", test_acc)
        self.log("test.auc", test_auc)

    def configure_optimizers(self):
        """
        Sets optimizer and scheduler.
        Must use strict equal to false because if check_val_n_epochs is > 1
        because val metrics not defined when scheduler is queried
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams.lr_eval,
            weight_decay=self.hparams.weight_decay_eval,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=int(10 / self.hparams.check_val_every_n_epoch),
            min_lr=self.hparams.lr * 0.0001,
        )
        return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "eval.val.loss",
                "strict": False,
            },
        }
