#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=ResCross-192-Frozen-Prenorm
#SBATCH --nodelist=zz
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-16:00:00
#SBATCH --output=/home/stympopper/bash/out/%x-%j.out
#SBATCH --error=/home/stympopper/bash/out/%x-%j.err


uname -a
nvidia-smi
cd MulModVM

# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=128 num_workers=8 persistent_workers=True use_cache=False use_transformer=True one_hot=False use_xtab=True wandb_name=Bob-LightTransformer-XTab-BN
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal pretrain=True max_epochs=100 batch_size=256 use_transformer=True num_workers=8
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=256 num_workers=8 persistent_workers=True use_cache=False use_transformer=False one_hot=True wandb_name=BoB-Baseline
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal checkpoint=/home/stympopper/MulModVM/runs/multimodal/Bob2-Transformer/checkpoint_last_epoch_99.ckpt max_epochs=100 batch_size=128 num_workers=26 wandb_name=BobTransFinetune1 use_cache=False use_transformer=True
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 batch_size=256 num_workers=8 persistent_workers=True use_cache=False use_transformer=True one_hot=False checkpoint=/home/stympopper/MulModVM/runs/multimodal/Bob-LightTransformer-XTab/checkpoint_last_epoch_45.ckpt wandb_name=Bob-XTab-Finetune
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal max_epochs=100 pretrain=True batch_size=256 num_workers=8 persistent_workers=True use_cache=False use_transformer=True one_hot=False use_xtab=True wandb_name=Bob-LightTransformer-XTab-Mask1
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal fusion=True max_epochs=100 batch_size=512 num_workers=16 tabular_embedding_dim=2048 tabular_model=mlp wandb_name=BobFusion-MLP-2048
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=tabular fusion=False evaluate=True max_epochs=100 batch_size=512 num_workers=8 tabular_embedding_dim=2048 tabular_model=mlp wandb_name=BobBaeseline-Tabular-MLP-2048
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=imaging_and_tabular fusion=True max_epochs=100 batch_size=512 num_workers=16 tabular_embedding_dim=2048 tabular_model=mlp wandb_name=BobFusion-MLP-2048_v2
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal fusion=True max_epochs=100 batch_size=512 num_workers=16 tabular_model=transformer tabular_embedding_dim=192 wandb_name=BoBFusion-Transformer-192_v2
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=multimodal fusion=True cross_fusion=False image_tokenization=False max_epochs=100 batch_size=128 num_workers=8 wandb_name=DummyConcat
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=imaging_and_tabular fusion=True cross_fusion=True image_tokenization=True max_epochs=100 batch_size=256 num_workers=8 wandb_name=CrossAtt-Img768
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=imaging_and_tabular fusion=True max_epochs=100 batch_size=256 num_workers=8 resnet_tokenization=True cross_fusion=True wandb_name=ResNet-CrossAtt-2048_dropout
# poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=imaging_and_tabular fusion=True max_epochs=100 batch_size=512 num_workers=8 tabular_model=mlp tabular_embedding_dim=2048 use_physical=False wandb_name=BoBConcat-NonPhysical-MLP
poetry run python run.py data_base=/home/stympopper/data/DVMdata/features/ datatype=imaging_and_tabular fusion=True max_epochs=100 cross_fusion=True resnet_tokenization=True batch_size=256 num_workers=8 fusion_core.first_prenormalization=True wandb_name=ResCross-192-Frozen-Pretrained-Prenorm