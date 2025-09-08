#!/bin/bash

# Parameters
#SBATCH --array=0-0%1
#SBATCH --error=${CHARLLM_ROOT}/CharLLM-PPT/results/download_llama_pile/preprocess/log-nemo-megatron-download_llama_pile.err
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --job-name=nemo-megatron-download_llama_pile
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --output=${CHARLLM_ROOT}/CharLLM-PPT/results/download_llama_pile/preprocess/log-nemo-megatron-download_llama_pile.out
#SBATCH --time=1:00:00

module load anaconda3
conda activate CharLLM-PPT

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts:${PYTHONPATH}

# command 1
srun --output ${CHARLLM_ROOT}/CharLLM-PPT/results/download_llama_pile/preprocess/log-nemo-megatron-download_llama_pile.out --error ${CHARLLM_ROOT}/CharLLM-PPT/results/download_llama_pile/preprocess/log-nemo-megatron-download_llama_pile.err bash -c "
  python3 -u ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/preprocess.py \
  data_config=download_llama_pile \
  cluster_type=bcm \
  launcher_scripts_path=${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts \
  data_dir=${CHARLLM_ROOT}/CharLLM-PPT/llama_data \
  the_pile_url=https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/ \
  file_numbers=0 \
  rm_downloaded=False \
  rm_extracted=False \
  tokenizer_type=None \
  tokenizer_library=sentencepiece \
  tokenizer_model=${CHARLLM_ROOT}/CharLLM-PPT/llama_data/llama_tokenizer.model \
  vocab_save_dir=None \
  merges_save_dir=None "
