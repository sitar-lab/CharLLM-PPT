#!/bin/bash

# Parameters
#SBATCH --error=${CHARLLM_ROOT}/CharLLM-PPT/results/lora/mixtral-8x7b/mixtral-8x7b_2_mbs2-h200.err
#SBATCH -C H200
#SBATCH --nodelist=${CHARLLM_H200_NODELIST}
#SBATCH --gpus-per-node=8
#SBATCH --partition=${CHARLLM_PARTITION}
#SBATCH --job-name=nemo-megatron-mixtral-8x7b
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=${CHARLLM_ROOT}/CharLLM-PPT/results/lora/mixtral-8x7b/mixtral-8x7b_2_mbs2-h200.out
#SBATCH --time=0-00:30:00



module load anaconda3
conda activate CharLLM-PPT

# setup
export TRANSFORMERS_OFFLINE=0
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0
export PYTHONPATH=${CHARLLM_ROOT}/CharLLM-PPT/NeMo:${PYTHONPATH}
export HYDRA_FULL_ERROR=1

# Hugging Face API Token
export HF_TOKEN=""

# Zeus Arguments
export ZEUS_CSV_PATH=${CHARLLM_ROOT}/CharLLM-PPT/results/lora/mixtral-8x7b/mixtral-8x7b_2_mbs2
export ZEUS_MONITOR_ENABLED=1
export ZEUS_MONITOR_SYS=1

mkdir -p $ZEUS_CSV_PATH

srun --output ${CHARLLM_ROOT}/CharLLM-PPT/results/lora/mixtral-8x7b/mixtral-8x7b_2_mbs2-h200.out --error ${CHARLLM_ROOT}/CharLLM-PPT/results/lora/mixtral-8x7b/mixtral-8x7b_2_mbs2-h200.err --mpi=pmix bash -c "
  CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NVTE_FWD_LAYERNORM_SM_MARGIN=\$(python3 ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_BWD_LAYERNORM_SM_MARGIN=\$(python3 ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ln_sm_margin) NVTE_UB_SPLIT_AG=\$(python3 ${CHARLLM_ROOT}/CharLLM-PPT/NeMo-Framework-Launcher/launcher_scripts/nemo_launcher/collections/conditional_cfgs.py name=get_ag_overlap fp8=False ) python3 -u ${CHARLLM_ROOT}/CharLLM-PPT/NeMo/examples/nlp/language_modeling/tuning/megatron_gpt_finetuning.py  \
  --config-path=${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/lora/mixtral-8x7b/main \
  --config-name=mixtral-8x7b_2_mbs2.yaml \
  "
