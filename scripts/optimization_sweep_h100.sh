#!/bin/bash
# filepath: /home/hice1/sgo38/scratch/micro-ae/CharLLM-PPT/scripts/optimization_sweep_h100.sh
# Description: Launch all sbatch scripts in a directory and its subdirectories recursively

# Function to show usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --mp         Launch only model parallelism experiments"
    echo "  --fsdp       Launch only FSDP experiments"
    echo "  --all        Launch both MP and FSDP experiments (default)"
    echo "  --help       Show this help message"
    exit 1
}

# Parse command line arguments
LAUNCH_MP=false
LAUNCH_FSDP=false
LAUNCH_ALL=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --mp)
            LAUNCH_MP=true
            LAUNCH_ALL=false
            shift
            ;;
        --fsdp)
            LAUNCH_FSDP=true
            LAUNCH_ALL=false
            shift
            ;;
        --all)
            LAUNCH_ALL=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# If no specific flags are set, default to launching all
if [ "$LAUNCH_MP" = false ] && [ "$LAUNCH_FSDP" = false ] && [ "$LAUNCH_ALL" = false ]; then
    LAUNCH_ALL=true
fi

# Deactivate conda if active
if command -v conda &> /dev/null; then
    echo "Deactivating conda environment..."
    conda deactivate
fi

MP_DIR=${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/optimization-sweep-h100
FSDP_DIR=${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/optimization-sweep-h100-fsdp

# Function to launch sbatch scripts recursively
launch_sbatch_recursive() {
    local dir="$1"
    local dir_type="$2"
    
    echo "Searching in directory: $dir ($dir_type)"
    
    if [ ! -d "$dir" ]; then
        echo "Warning: Directory $dir does not exist, skipping..."
        return
    fi
    
    # Launch all sbatch scripts in current directory
    for script in "$dir"/*.sh; do
        if [ -f "$script" ]; then
            echo "Submitting $script..."
            sbatch "$script"
        fi
    done
    
    # Recursively search subdirectories
    for subdir in "$dir"/*; do
        if [ -d "$subdir" ]; then
            launch_sbatch_recursive "$subdir" "$dir_type"
        fi
    done
}

echo "Launch configuration:"
echo "  MP: $LAUNCH_MP"
echo "  FSDP: $LAUNCH_FSDP"
echo "  ALL: $LAUNCH_ALL"
echo ""

# Launch based on flags
if [ "$LAUNCH_ALL" = true ] || [ "$LAUNCH_MP" = true ]; then
    echo "=== Launching Model Parallelism experiments ==="
    launch_sbatch_recursive "$MP_DIR" "Model Parallelism"
fi

if [ "$LAUNCH_ALL" = true ] || [ "$LAUNCH_FSDP" = true ]; then
    echo "=== Launching FSDP experiments ==="
    launch_sbatch_recursive "$FSDP_DIR" "FSDP"
fi

echo "Done launching all sbatch scripts."