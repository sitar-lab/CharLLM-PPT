#!/bin/bash
# This script launches the test for data preparation pipeline and GPT training.

# Submit the first script and capture the job ID
job_id=$(sbatch ${CHARLLM_ROOT}/CharLLM-PPT/scripts/test/test_dataprep.sbatch | awk '{print $4}')
echo "Submitted test_dataprep.sbatch with Job ID: $job_id"

# Submit the second script with a dependency on the first script
sbatch --dependency=afterok:$job_id ${CHARLLM_ROOT}/CharLLM-PPT/scripts/test/test_gpt.sbatch
echo "Submitted test_gpt.sbatch with dependency on Job ID: $job_id"