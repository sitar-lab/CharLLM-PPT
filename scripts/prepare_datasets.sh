# Deactivate conda if active
if command -v conda &> /dev/null; then
    echo "Deactivating conda environment..."
    conda deactivate
fi

echo "Launching Pile dataset download jobs..."
sbatch ${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/pile/gpt3/gpt-download.sh
sbatch ${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/pile/llama3/llama-download.sh
sbatch ${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/pile/mixtral_8x7b/mixtral8x7b-download.sh
