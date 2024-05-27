#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=top
#SBATCH --job-name=BoBW-Baseline
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err


uname -a
nvidia-smi
cd MulModVM

poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=256 num_workers=8 persistent_workers=True use_cache=False use_transformer=True one_hot=False use_xtab=True wandb_name=Bob-LightTransformer-XTab
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=256 num_workers=8 persistent_workers=True use_cache=False use_transformer=False one_hot=True wandb_name=BoB-Baseline
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal checkpoint=/home/stympopper/MulModVM/runs/multimodal/Bob2-Transformer/checkpoint_last_epoch_99.ckpt max_epochs=100 batch_size=128 num_workers=26 wandb_name=BobTransFinetune1 use_cache=False use_transformer=True
