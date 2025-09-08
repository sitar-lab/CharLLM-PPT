#!/bin/bash
# filepath: ${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/pile/mixtral_8x7b/combined_mixtral_pipeline.sh

# Parameters for the combined job
#SBATCH --array=0-0%1
#SBATCH -C H200
#SBATCH --error=${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/combined/log-nemo-megatron-download_mixtral_pile.err
#SBATCH --gpus-per-node=4
#SBATCH --job-name=mixtral_pile_pipeline
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=${CHARLLM_ROOT}/CharLLM-PPT/results/download_mixtral_pile/combined/log-nemo-megatron-download_mixtral_pile.out
#SBATCH --time=2:00:00

module load anaconda3
conda activate CharLLM-PPT

# setup
export CHARLLM_ROOT=${CHARLLM_ROOT}
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts:${PYTHONPATH}
export HF_TOKEN=${HF_TOKEN}

echo "Starting Mixtral Pile data pipeline..."

# Step 1: Download
echo "Step 1: Downloading Mixtral data..."
srun bash -c "
  CHARLLM_ROOT=${CHARLLM_ROOT} \
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

if [ $? -ne 0 ]; then
    echo "ERROR: Download step failed"
    exit 1
fi

echo "Step 2: Extracting Mixtral data..."
srun bash -c "
  CHARLLM_ROOT=${CHARLLM_ROOT} \
  python3 -u ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/extract.py \
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

if [ $? -ne 0 ]; then
    echo "ERROR: Extract step failed"
    exit 1
fi

echo "Step 3: Preprocessing Mixtral data..."
srun bash -c "
  CHARLLM_ROOT=${CHARLLM_ROOT} \
  python3 -u ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/dataprep_scripts/pile_dataprep/preprocess.py \
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

if [ $? -ne 0 ]; then
    echo "ERROR: Preprocess step failed"
    exit 1
fi

echo "Mixtral pipeline completed successfully!"