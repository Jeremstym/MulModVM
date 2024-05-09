#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=BoBW-attention
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1-16:00:00
#SBATCH --output=/home/stympopper/MulModVM/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/MulModVM/bash/out/%x-%j.err


uname -a
nvidia-smi
cd MulModVM

poetry shell
python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=tabular max_epochs=100

