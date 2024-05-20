#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=BoBW-attention
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err


uname -a
nvidia-smi
cd MulModVM

# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=128 num_workers=28 use_cache=False use_transformer=True one_hot=False wandb_name=Bob2-Transformer
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=128 num_workers=28 use_cache=False use_transformer=True wandb_name=Bob1-Transformer
python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal checkpoint=/home/stympopper/MulModVM/runs/multimodal/Bob2-Transformer/checkpoint_last_epoch_99.ckpt max_epochs=100 batch_size=128 num_workers=26 wandb_name=BobTransFinetune1 use_cache=False use_transformer=True
