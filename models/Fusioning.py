from typing import Tuple

import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

from models.TabularEncoder import TabularEncoder
from models.TabularTransformer import TabularTransformer
from models.ImagingModel import ImagingModel
from models.ImageTokenizer import ViTTokenizer
from models.FusionCore import FusionCoreCrossAtt, FusionCoreConcat


class Fusion(pl.LightningModule):
    def __init__(self, hparams, dataset=None, use_projection: bool = False):
        super().__init__()
        self.save_hyperparameters(hparams)

        # Initialize tabular model
        if hparams.tabular_model == "transformer":
            assert dataset is not None, "Dataset must be provided for transformer models"
            cat_mask = dataset.get_cat_mask()
            self.cat_mask = cat_mask
            num_cont = dataset.get_number_of_numerical_features()
            cat_card = dataset.get_cat_card()
            cat_cardinalities = cat_card.tolist()
            assert isinstance(
                self.hparams.tabular_tokenizer, DictConfig
            ), "Tabular tokenizer must be provided for transformer models"
            self.tabular_tokenizer = hydra.utils.instantiate(
                self.hparams.tabular_tokenizer,
                cat_cardinalities=cat_cardinalities,
                n_num_features=num_cont,
            )

        assert (
            self.hparams.datatype == "imaging_and_tabular"
            or self.hparams.datatype == "multimodal"
        ), "Fusion model must be imaging_and_tabular or multimodal"
        
        self.use_projection = use_projection
        
        # Intialize fusion core
        if self.hparams.cross_fusion:
            self.fusion_core = FusionCoreCrossAtt(self.hparams)
            self.hidden_size = self.hparams.hidden_size
        else:
            self.fusion_core = FusionCoreConcat(self.hparams)
            self.hidden_size = self.hparams.embedding_dim

        # Initialize imaging model
        if self.hparams.image_tokenization:
            assert not self.hparams.use_vit, "ViT model not supported with tokenization because it takes images as input."
            self.imaging_tokenizer = ViTTokenizer(self.hparams)
        elif self.hparams.use_vit:
            assert self.hparams.model == "vit-b-32", "Only vit-b-32 model supported."
            self.imaging_model = ImagingModel(self.hparams)
        else:
            self.imaging_model = ImagingModel(self.hparams)

        # Initialize tabular encoders
        if self.hparams.tabular_model == "transformer":
            self.encoder_tabular = TabularTransformer(self.hparams)
            if self.hparams.use_xtab:
                self.load_pretrained_xtab()
        elif self.hparams.tabular_model == "mlp":
            self.encoder_tabular = TabularEncoder(self.hparams)
        if self.use_projection:
            self.tab_head = nn.Linear(
                self.hparams.tabular_embedding_dim, self.hparams.projection_dim
            )
            self.im_head = nn.Linear(
                self.hparams.embedding_dim, self.hparams.projection_dim
            )
            self.head = nn.Linear(self.hparams.projection_dim * 2, self.hparams.num_classes)
        else:
            if hparams.cross_fusion:
                head_input_dim = self.hidden_size
            else:
                head_input_dim = self.hidden_size + self.hparams.tabular_embedding_dim
            self.head = nn.Linear(head_input_dim, self.hparams.num_classes)

        # Metrics
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
        self.best_val_score = 0

        # Loss
        self.criterion = torch.nn.CrossEntropyLoss()

        # print all model
        if self.hparams.tabular_model == "transformer":
            print(self.tabular_tokenizer)
        if self.hparams.image_tokenization:
            print(self.imaging_tokenizer)
        else:
            print(self.imaging_model.encoder)
        print(self.encoder_tabular)
        if use_projection:
            print(self.im_head)
            print(self.tab_head)
        print(self.head)

    def tokenize_tabular(self, x: torch.Tensor) -> torch.Tensor:
        x_num = x[:, ~self.cat_mask]
        x_cat = x[:, self.cat_mask].type(torch.int64)
        x = self.tabular_tokenizer(x_num=x_num, x_cat=x_cat)
        return x

    # def encode_tabular(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    #     x = self.encoder_tabular(x, mask)
    #     return x

    def encode_imaging(self, x: torch.Tensor) -> torch.Tensor:
        if self.hparams.image_tokenization:
            x = self.imaging_tokenizer(x) 
        elif self.imaging_model.bolt_encoder:
            x = self.imaging_model.encoder(x)[0]
        else:
            x = self.imaging_model.encoder(x).squeeze()
        return x

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x_im = self.encode_imaging(x[0])  # only keep the encoder output
        print(f"Imaging model output shape: {x_im.shape}")
        if self.hparams.tabular_model == "transformer":
            x_tokens_tab = self.tokenize_tabular(x[1])
            x_tab = self.encoder_tabular(x_tokens_tab).squeeze()
            print(f"Tabular model output shape: {x_tab.shape}")
        else:
            x_tab = self.encoder_tabular(x[1])
        if self.hparams.cross_fusion:
            x = self.fusion_core(x_im, x_tab)
            print(f"Fusion core output shape: {x.shape}")
            x = x[:, -1, :]
        else:
            x_tab = x_tab[:, -1, :]
            x = torch.cat([x_im, x_tab], dim=1)
        # if self.use_projection:
        #     x_im = self.im_head(x_im)
        #     x_tab = self.tab_head(x_tab)
        # x = torch.cat([x_im, x_tab], dim=1)
        x = self.head(x)
        print(f"Output shape: {x.shape}")
        raise Exception("Stop here")
        return x

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

        self.log("fusion.train.loss", loss, on_epoch=True, on_step=False)
        return loss

    def training_epoch_end(self, _) -> None:
        """
        Training epoch end
        """
        self.log(
            "fusion.train.acc",
            self.acc_train,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.acc_train,
        )

        self.log(
            "fusion.train.auc",
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

        self.log("fusion.val.loss", loss)

    def validation_epoch_end(self, _) -> None:
        """
        Validation epoch end
        """
        if self.trainer.sanity_checking:
            return

        epoch_val_acc = self.acc_val.compute()
        epoch_val_auc = self.auc_val.compute()

        self.log(
            "fusion.val.acc",
            epoch_val_acc,
            on_epoch=True,
            on_step=False,
            metric_attribute=self.acc_val,
        )

        self.log(
            "fusion.val.auc",
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
        if self.hparams.tabular_model == "transformer" and self.hparams.image_tokenization:
            optimizer = torch.optim.Adam(
                [
                    # {"params": self.imaging_model.parameters()},
                    {"params": self.imaging_tokenizer.parameters()},
                    # {"params": self.im_head.parameters()},
                    {"params": self.tabular_tokenizer.parameters()},
                    {"params": self.encoder_tabular.parameters()},
                    # {"params": self.tab_head.parameters()},
                    {"params": self.fusion_core.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=self.hparams.lr_eval,
                weight_decay=self.hparams.weight_decay_eval,
            )
        elif self.hparams.tabular_model == "transformer":
            optimizer = torch.optim.Adam(
                [
                    {"params": self.imaging_model.parameters()},
                    {"params": self.tabular_tokenizer.parameters()},
                    {"params": self.encoder_tabular.parameters()},
                    # {"params": self.im_head.parameters()},
                    # {"params": self.tab_head.parameters()},
                    {"params": self.fusion_core.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=self.hparams.lr_eval,
                weight_decay=self.hparams.weight_decay_eval,
            )
        elif self.hparams.tabular_model == "mlp":
            optimizer = torch.optim.Adam(
                [
                    {"params": self.imaging_model.parameters()},
                    # {"params": self.im_head.parameters()},
                    {"params": self.encoder_tabular.parameters()},
                    # {"params": self.tab_head.parameters()},
                    {"params": self.head.parameters()},
                ],
                lr=self.hparams.lr_eval,
                weight_decay=self.hparams.weight_decay_eval,
            )
        else:   
            raise ValueError("Tabular model unknown. Must be 'transformer' or 'mlp'.")
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

    def load_pretrained_xtab(self) -> None:
        """
        Can load tabular encoder with pretrained weights from XTab foundation model
        """
        loaded_chkpt = torch.load(self.hparams.xtab_path, map_location=self.device)
        self.encoder_tabular.load_state_dict(loaded_chkpt, strict=False) # no state_dict key needed as it is the whole state_dict
        learned_layer = [layer for layer in self.encoder_tabular.state_dict()]
        xtab_layer = [layer for layer in loaded_chkpt.keys()]
        intersection = set(learned_layer).intersection(set(xtab_layer))
        assert len(intersection) > 0, "No layers in common between learned model and XTab model"
        print(f"Loaded XTab model with layers: {intersection}")
        return