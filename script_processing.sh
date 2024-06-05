#!/bin/bash
#SBATCH --partition=hard
#SBATCH --nodelist=top
#SBATCH --job-name=BoB_DataProcessing
#SBATCH --time=1-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err


uname -a
nvidia-smi
cd MulModVM/datasets_processor

poetry run python dataset_tool.py --source=/home/stympopper/data/DVMdata/features/train_paths_all_views.pt --dest=/home/stympopper/data/DVMdata/readyTrainLabeled --resolution=128x128 --normalize=True --labels=/home/stympopper/data/DVMdata/features/dataset_train.json
poetry run python dataset_tool.py --source=/home/stympopper/data/DVMdata/features/val_paths_all_views.pt --dest=/home/stympopper/data/DVMdata/readyValLabeled --resolution=128x128 --normalize=True --labels=/home/stympopper/data/DVMdata/features/dataset_val.json