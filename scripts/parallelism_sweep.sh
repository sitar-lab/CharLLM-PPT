#!/bin/bash
# Description: Launch all sbatch scripts in a directory and its subdirectories recursively that contain "01000" in filename

MP_DIR=${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/optimization-sweep-h200
FSDP_DIR=${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/optimization-sweep-h200-fsdp

# Function to launch sbatch scripts recursively
launch_sbatch_recursive() {
    local dir="$1"
    
    echo "Searching in directory: $dir"
    
    # Launch all sbatch scripts in current directory that contain "01000"
    for script in "$dir"/*.sh; do
        if [ -f "$script" ]; then
            # Check if filename contains "01000"
            if [[ "$(basename "$script")" == *"01000"* ]]; then
                echo "Submitting $script..."
                sbatch "$script"
                # Optional: Add small delay between submissions
                # sleep 1
            # else
            #     echo "Skipping $script (doesn't contain '01000')"
            fi
        fi
    done
    
    # Recursively search subdirectories
    for subdir in "$dir"/*; do
        if [ -d "$subdir" ]; then
            launch_sbatch_recursive "$subdir"
        fi
    done
}

echo "Launch configuration:"
echo "  MP: $LAUNCH_MP"
echo "  FSDP: $LAUNCH_FSDP"
echo "  ALL: $LAUNCH_ALL"
echo ""

# Launch based on flags
echo "=== Launching Model Parallelism experiments ==="
launch_sbatch_recursive "$MP_DIR" "Model Parallelism"


echo "=== Launching FSDP experiments ==="
launch_sbatch_recursive "$FSDP_DIR" "FSDP"

echo "Done launching all sbatch scripts."