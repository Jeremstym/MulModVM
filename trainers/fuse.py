import os

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from torch.utils.data.sampler import WeightedRandomSampler

from datasets_processor.ImageFastDataset import ImageFastDataset
from datasets_processor.TabularDataset import TabularDataset
from datasets_processor.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Evaluator import Evaluator
from models.Evaluator_regression import Evaluator_Regression
from utils.utils import (
    grab_arg_from_checkpoint,
    grab_hard_eval_image_augmentations,
    grab_wids,
    create_logdir,
)


def load_datasets(hparams):
    if hparams.datatype == "imaging_or_tabular" or hparams.datatype == "multimodal":
        train_dataset = ImageFastDataset(
            hparams.data_train_eval_imaging,
            hparams.labels_train_eval_imaging,
            hparams.delete_segmentation,
            hparams.eval_train_augment_rate,
            grab_arg_from_checkpoint(hparams, "img_size"),
            target=hparams.target,
            train=True,
            live_loading=hparams.live_loading,
            task=hparams.task,
        )
        val_dataset = ImageFastDataset(
            hparams.data_val_eval_imaging,
            hparams.labels_val_eval_imaging,
            hparams.delete_segmentation,
            hparams.eval_train_augment_rate,
            grab_arg_from_checkpoint(hparams, "img_size"),
            target=hparams.target,
            train=False,
            live_loading=hparams.live_loading,
            task=hparams.task,
        )
    else:
        raise Exception(
            "argument dataset must be set to multimodal or imaging_and_tabular for the Fusion model"
        )
    return train_dataset, val_dataset


def fuse(hparams, wandb_logger):
    """
    Evaluates trained contrastive models.

    IN
    hparams:      All hyperparameters
    wandb_logger: Instantiated weights and biases logger
    """
    pl.seed_everything(hparams.seed)

    train_dataset, val_dataset = load_datasets(hparams)

    drop = (len(train_dataset) % hparams.batch_size) == 1

    sampler = None
    if hparams.weights:
        print("Using weighted random sampler(")
        weights_list = [hparams.weights[int(l)] for l in train_dataset.labels]
        sampler = WeightedRandomSampler(
            weights=weights_list, num_samples=len(weights_list), replacement=True
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=hparams.num_workers,
        persistent_workers=hparams.persistent_workers,
        drop_last=drop,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hparams.batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        persistent_workers=hparams.persistent_workers,
    )

    if hparams.datatype == "imaging_or_tabular" or hparams.datatype == "multimodal":
        model = Fusion(hparams, dataset=train_dataset)
    else:
        raise Exception(
            "argument dataset must be set to imaging_or_tabular or multimodal for the Fusion model"
        )

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor=f"fusion.val.{hparams.eval_metric}",
            mode=mode,
            filename=f"checkpoint_best_{hparams.eval_metric}",
            dirpath=logdir,
        )
    )
    callbacks.append(
        EarlyStopping(
            monitor=f"fusion.val.{hparams.eval_metric}",
            min_delta=0.0002,
            patience=int(10 * (1 / hparams.val_check_interval)),
            verbose=False,
            mode=mode,
        )
    )
    if hparams.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    log_dir = create_logdir("fusion", hparams.logdir)

    trainer = Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=hparams.epochs,
        check_val_every_n_epoch=hparams.check_val_every_n_epoch,
        val_check_interval=hparams.val_check_interval,
        limit_train_batches=hparams.limit_train_batches,
        limit_val_batches=hparams.limit_val_batches,
        limit_test_batches=hparams.limit_test_batches,
        enable_progress_bar=hparams.enable_progress_bar,
        profiler=hparams.profiler,
    )

    trainer.fit(model, train_loader, val_loader)

    wandb_logger.log_metrics({f"best.val.{hparams.eval_metric}": model.best_val_score})

    return model
