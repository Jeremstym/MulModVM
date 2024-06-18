from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets_processor.ImageDataset import ImageDataset
from datasets_processor.ImageFastDataset import ImageFastDataset
from datasets_processor.ImagingAndTabularDataset import ImagingAndTabularDataset
from datasets_processor.TabularDataset import TabularDataset
from models.Evaluator import Evaluator
from models.Fusioning import Fusion
from utils.utils import grab_arg_from_checkpoint


def test(hparams, wandb_logger=None):
    """
    Tests trained models.

    IN
    hparams:      All hyperparameters
    """
    pl.seed_everything(hparams.seed)

    if hparams.datatype == "imaging":
        test_dataset = ImageFastDataset(
            data_path=hparams.data_fast_test_imaging,
            name="imaging_test",
            labels_path_short=hparams.labels_test_eval_imaging,
            delete_segmentation=hparams.delete_segmentation,
            train_augment_rate=0.0,
            img_size=grab_arg_from_checkpoint(hparams, "img_size"),
            target=hparams.target,
            use_labels=True,
            train=False,
            live_loading=hparams.live_loading,
        )

        print(test_dataset.transform_val.__repr__())
    elif hparams.datatype == "tabular":
        test_dataset = TabularDataset(
            hparams.data_test_eval_tabular,
            hparams.labels_test_eval_tabular,
            hparams.eval_one_hot,
            hparams.field_lengths_tabular,
        )
        hparams.input_size = test_dataset.get_input_size()
    elif hparams.datatype == "multimodal" or hparams.datatype == "imaging_and_tabular":
        test_dataset = ImagingAndTabularDataset(
            data_path_imaging=hparams.data_fast_test_imaging,
            data_path_tabular=hparams.data_test_eval_tabular,
            delete_segmentation=hparams.delete_segmentation,
            eval_train_augment_rate=hparams.eval_train_augment_rate,
            labels_path=hparams.labels_test_eval_imaging,
            field_lengths_tabular=hparams.field_lengths_tabular,
            eval_one_hot=hparams.eval_one_hot,
            img_size=hparams.img_size,
            target=hparams.target,
            tabular_model=hparams.tabular_model,
            train=False,
            live_loading=hparams.live_loading,
        )
        hparams.input_size = test_dataset.get_input_size()
    else:
        raise Exception(
            "argument dataset must be set to imaging, tabular or multimodal"
        )

    drop = (len(test_dataset) % hparams.batch_size) == 1

    test_loader = DataLoader(
        test_dataset,
        num_workers=hparams.num_workers,
        batch_size=hparams.batch_size,
        pin_memory=True,
        shuffle=False,
        drop_last=drop,
        persistent_workers=hparams.persistent_workers,
    )

    hparams.dataset_length = len(test_loader)

    # model = Evaluator(hparams)
    model = Fusion(hparams, dataset=test_dataset)
    model.freeze()
    trainer = Trainer.from_argparse_args(hparams, gpus=1, logger=wandb_logger)
    trainer.test(model, test_loader, ckpt_path=hparams.checkpoint)
    # trainer.test(model, test_loader)
