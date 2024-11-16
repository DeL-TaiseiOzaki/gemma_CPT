#!/bin/bash
#SBATCH --job-name=gemma-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=24:00:00
#SBATCH --mem=500GB
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err

# 環境変数の設定
export MASTER_PORT=29500
export MASTER_ADDR=$(hostname)

# conda環境のアクティベート（環境に応じて変更）
source ~/.bashrc
conda activate llm_env

# WandBのセットアップ（必要に応じて）
wandb login YOUR_WANDB_KEY

# DeepSpeedでの学習開始
deepspeed --num_gpus=8 \
    src/train_deepspeed.py \
    --train_config ./configs/train_configs/train_base.yaml