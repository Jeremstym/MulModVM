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
from datasets_processor.ContrastiveFastImagingAndTabularDataset import (
    ContrastiveFastImagingAndTabularDataset,
)
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

from models.Fusioning import Fusion



def load_datasets(hparams):
    if hparams.datatype == "multimodal":
        train_dataset = ContrastiveFastImagingAndTabularDataset(
            data_path_imaging=hparams.data_fast_train_imaging,
            delete_segmentation=hparams.delete_segmentation,
            augmentation_rate=hparams.augmentation_rate,
            data_path_tabular=hparams.data_train_tabular,
            corruption_rate=hparams.corruption_rate,
            field_lengths_tabular=hparams.field_lengths_tabular,
            one_hot_tabular=hparams.eval_one_hot,
            labels_path=hparams.labels_train,
            img_size=hparams.img_size,
            target=hparams.target,
            missing_values=hparams.missing_values,
            tabular_model=hparams.tabular_model,
            use_labels=True,
            max_size=None,
        )
        val_dataset = ContrastiveFastImagingAndTabularDataset(
            data_path_imaging=hparams.data_fast_val_imaging,
            delete_segmentation=hparams.delete_segmentation,
            augmentation_rate=hparams.augmentation_rate,
            data_path_tabular=hparams.data_val_tabular,
            corruption_rate=hparams.corruption_rate,
            field_lengths_tabular=hparams.field_lengths_tabular,
            one_hot_tabular=hparams.eval_one_hot,
            labels_path=hparams.labels_val,
            img_size=hparams.img_size,
            target=hparams.target,
            missing_values=hparams.missing_values,
            tabular_model=hparams.tabular_model,
            use_labels=True,
            max_size=None,
        )
        hparams.input_size = train_dataset.get_input_size()
    elif hparams.datatype == "imaging_and_tabular":
        train_dataset = ImagingAndTabularDataset(
            data_path_imaging=hparams.data_fast_train_imaging,
            data_path_tabular=hparams.data_train_tabular,
            delete_segmentation=hparams.delete_segmentation,
            eval_train_augment_rate=hparams.eval_train_augment_rate,
            labels_path=hparams.labels_train,
            field_lengths_tabular=hparams.field_lengths_tabular,
            eval_one_hot=hparams.eval_one_hot,
            img_size=hparams.img_size,
            target=hparams.target,
            # missing_values=hparams.missing_values,
            tabular_model=hparams.tabular_model,
            train=True,
            live_loading=hparams.live_loading,
        )
        val_dataset = ImagingAndTabularDataset(
            data_path_imaging=hparams.data_fast_val_imaging,
            data_path_tabular=hparams.data_val_tabular,
            delete_segmentation=hparams.delete_segmentation,
            eval_train_augment_rate=hparams.eval_train_augment_rate,
            labels_path=hparams.labels_val,
            field_lengths_tabular=hparams.field_lengths_tabular,
            eval_one_hot=hparams.eval_one_hot,
            img_size=hparams.img_size,
            target=hparams.target,
            # missing_values=hparams.missing_values,
            tabular_model=hparams.tabular_model,
            train=False,
            live_loading=hparams.live_loading,
        )
        hparams.input_size = train_dataset.get_input_size()
    else:
        raise Exception(
            f"argument dataset must be set to multimodal or imaging_and_tabular AND GOT {hparams.datatype} instead"
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
            f"argument dataset must be set to imaging_or_tabular or multimodal for the Fusion model, got {hparams.datatype} instead"
        )
    logdir = create_logdir('eval', hparams.resume_training, wandb_logger)
    
    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor=f"fusion_val_{hparams.eval_metric}",
            mode="max",
            filename=f"checkpoint_best_{hparams.eval_metric}",
            dirpath=logdir,
        )
    )
    # callbacks.append(
    #     EarlyStopping(
    #         monitor=f"fusion_val_{hparams.eval_metric}",
    #         min_delta=0.0002,
    #         patience=int(10 * (1 / hparams.val_check_interval)),
    #         verbose=False,
    #         mode="max",
    #     )
    # )
    if hparams.use_wandb:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    lr_monitor = LearningRateMonitor(logging_interval="epoch")


    trainer = Trainer.from_argparse_args(
        hparams,
        accelerator="gpu",
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
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
