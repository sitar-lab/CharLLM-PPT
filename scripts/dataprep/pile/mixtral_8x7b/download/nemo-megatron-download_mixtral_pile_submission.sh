#!/bin/bash

# Parameters
#SBATCH --array=0-0%1
#SBATCH --exclusive
#SBATCH -C L40S
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nemo-megatron-download_mixtral_pile
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --output=${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/download/log-nemo-megatron-download_mixtral_pile.out
#SBATCH --error=${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/download/log-nemo-megatron-download_mixtral_pile.out
#SBATCH --time=1:00:00

module load anaconda3
conda activate CharLLM-PPT

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts:${PYTHONPATH}

srun --output ${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/download/log-nemo-megatron-download_mixtral_pile.out --error ${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/download/log-nemo-megatron-download_mixtral_pile.out bash -c "
  python3 -u ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/download.py \
  data_config=download_mixtral_pile \
  cluster_type=bcm \
  launcher_scripts_path=${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts \
  data_dir=${CHARLLM_ROOT}/CharLLM-PPT/mixtral_data \
  the_pile_url=https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/ \
  file_numbers=0 \
  rm_downloaded=False \
  rm_extracted=False \
  tokenizer_type=mistralai/Mixtral-8x7B-v0.1 \
  tokenizer_library=huggingface \
  tokenizer_model=None \
  vocab_save_dir=None \
  merges_save_dir=None "
