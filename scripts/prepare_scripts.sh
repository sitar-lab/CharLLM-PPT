########################################################################

# Fill in the following environment variables before running this script
# Environment variables
export CHARLLM_ROOT= # Root directory of the CharLLM-PPT project
export HF_TOKEN= # Hugging Face token for accessing datasets and models
export HF_HOME= # Hugging Face home directory for caching datasets and models

# SBATCH configuration
export GPU_TYPE=H200 # Configure accordingly
export GPUS_PER_NODE=8
export MEM_PER_NODE=0 # 0 means the maximum available memory
export NUM_NODES=4
export NTASKS_PER_NODE=8
export CPUS_PER_TASK=8
export CHARLLM_PARTITION=
export CHARLLM_QOS=
export CHARLLM_H200_NODELIST=
export TIME_LIMIT=0-10:00:00 # 10 hours per run, adjust as needed

########################################################################

# Resulting environment variables
echo "Resulting environment variables:"
echo "CHARLLM_ROOT: ${CHARLLM_ROOT:-"(not set)"}"
echo "HF_TOKEN: ${HF_TOKEN:-"(not set)"}"
echo "HF_HOME: ${HF_HOME:-"(not set)"}"

# Resulting SBATCH configuration
echo "Resulting SBATCH configuration:"
echo "#SBATCH -C ${GPU_TYPE:-"(not set)"}"
echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE:-"(not set)"}"
echo "#SBATCH --mem=${MEM_PER_NODE:-"(not set)"}"
echo "#SBATCH --nodes=${NUM_NODES:-"(not set)"}"
echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE:-"(not set)"}"
echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK:-"(not set)"}"
echo "#SBATCH --partition=${CHARLLM_PARTITION:-"(not set)"}"
echo "#SBATCH --qos=${CHARLLM_QOS:-"(not set)"}"
echo "#SBATCH --nodelist=${CHARLLM_H200_NODELIST:-"(not set)"}"
echo "#SBATCH --time=${TIME_LIMIT:-"(not set)"}"


# Ignore the following environment variables; will be auto-set in the script
export CHARLLM_FSDP=0
export CHARLLM_UNEVEN_PIPELINE=0

# Step 1: Go through all the files under ${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/ then replace the following words:
# ${CHARLLM_ROOT} with actual value
# ${CHARLLM_H200_NODELIST} with actual value
echo "Step 1: Replacing placeholders in all script files..."

find "${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/" -mindepth 2 -type f \( -name "*.sh" -o -name "*.yaml" \) -print0 | while IFS= read -r -d '' file; do
    echo "Processing file: $file"
    
    # Replace ${CHARLLM_ROOT} with actual value
    sed -i "s|\${CHARLLM_ROOT}|${CHARLLM_ROOT}|g" "$file"

    # Replace ${HF_TOKEN} with actual value
    sed -i "s|\${HF_TOKEN}|${HF_TOKEN}|g" "$file"

    # Replace ${CHARLLM_PARTITION} with actual value or remove line if not set
    if [ -n "${CHARLLM_PARTITION}" ]; then
        sed -i "s|\${CHARLLM_PARTITION}|${CHARLLM_PARTITION}|g" "$file"
    else
        sed -i '/\${CHARLLM_PARTITION}/d' "$file"
    fi
    
    # Replace ${CHARLLM_QOS} with actual value or remove line if not set
    if [ -n "${CHARLLM_QOS}" ]; then
        sed -i "s|\${CHARLLM_QOS}|${CHARLLM_QOS}|g" "$file"
    else
        sed -i '/\${CHARLLM_QOS}/d' "$file"
    fi
    
    # Replace ${CHARLLM_H200_NODELIST} with actual value or remove line if not set
    if [ -n "${CHARLLM_H200_NODELIST}" ]; then
        sed -i "s|\${CHARLLM_H200_NODELIST}|${CHARLLM_H200_NODELIST}|g" "$file"
    else
        sed -i '/\${CHARLLM_H200_NODELIST}/d' "$file"
    fi

done

find "${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/" -mindepth 2 -type f \( -name "*.sh" -o -name "*.yaml" \) -print0 | while IFS= read -r -d '' file; do
    echo "Processing file: $file"
    
    # Replace ${CHARLLM_ROOT} with actual value
    sed -i "s|\${CHARLLM_ROOT}|${CHARLLM_ROOT}|g" "$file"

    # Replace ${HF_TOKEN} with actual value
    sed -i "s|\${HF_TOKEN}|${HF_TOKEN}|g" "$file"

    # Replace ${CHARLLM_PARTITION} with actual value or remove line if not set
    if [ -n "${CHARLLM_PARTITION}" ]; then
        sed -i "s|\${CHARLLM_PARTITION}|${CHARLLM_PARTITION}|g" "$file"
    else
        sed -i '/\${CHARLLM_PARTITION}/d' "$file"
    fi
    
    # Replace ${CHARLLM_QOS} with actual value or remove line if not set
    if [ -n "${CHARLLM_QOS}" ]; then
        sed -i "s|\${CHARLLM_QOS}|${CHARLLM_QOS}|g" "$file"
    else
        sed -i '/\${CHARLLM_QOS}/d' "$file"
    fi
    
    # Replace ${CHARLLM_H200_NODELIST} with actual value or remove line if not set
    if [ -n "${CHARLLM_H200_NODELIST}" ]; then
        sed -i "s|\${CHARLLM_H200_NODELIST}|${CHARLLM_H200_NODELIST}|g" "$file"
    else
        sed -i '/\${CHARLLM_H200_NODELIST}/d' "$file"
    fi

done

find "${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/" -mindepth 1 -type f \( -name "*.py" -o -name "*.sh" \) -print0 | while IFS= read -r -d '' file; do
    echo "Processing file: $file"
    
    # Replace ${CHARLLM_ROOT} with actual value
    sed -i "s|\${CHARLLM_ROOT}|${CHARLLM_ROOT}|g" "$file"

    # Replace ${CHARLLM_PARTITION} with actual value or remove line if not set
    if [ -n "${CHARLLM_PARTITION}" ]; then
        sed -i "s|\${CHARLLM_PARTITION}|${CHARLLM_PARTITION}|g" "$file"
    else
        sed -i '/\${CHARLLM_PARTITION}/d' "$file"
    fi
    
    # Replace ${CHARLLM_QOS} with actual value or remove line if not set
    if [ -n "${CHARLLM_QOS}" ]; then
        sed -i "s|\${CHARLLM_QOS}|${CHARLLM_QOS}|g" "$file"
    else
        sed -i '/\${CHARLLM_QOS}/d' "$file"
    fi
    
    # Replace ${CHARLLM_H200_NODELIST} with actual value or remove line if not set
    if [ -n "${CHARLLM_H200_NODELIST}" ]; then
        sed -i "s|\${CHARLLM_H200_NODELIST}|${CHARLLM_H200_NODELIST}|g" "$file"
    else
        sed -i '/\${CHARLLM_H200_NODELIST}/d' "$file"
    fi
    
done



echo "Step 1 completed."
echo ""

# Step 2: Go through all the files under ${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/ then replace all SBATCH lines (except --error and --output)
echo "Step 2: Replacing SBATCH lines in scripts..."

find "${CHARLLM_ROOT}/CharLLM-PPT/scripts/experiments/" -mindepth 2 -type f \( -name "*.sh" -o -name "*.py" \) -print0 | while IFS= read -r -d '' file; do
    if grep -q "module load anaconda3" "$file"; then
        echo "Found 'module load anaconda3' in: $file"
        
        # Check if file path contains "-fsdp" to determine FSDP value
        if [[ "$file" == *"-fsdp"* ]]; then
            FSDP_VALUE=1
            echo "  Setting CHARLLM_FSDP=1 (detected -fsdp in path)"
        else
            FSDP_VALUE=${CHARLLM_FSDP}
            echo "  Setting CHARLLM_FSDP=${CHARLLM_FSDP} (default value)"
        fi
        
        # Create a temporary file with the modifications
        temp_file=$(mktemp)

        # Process the file line by line
        while IFS= read -r line; do
            # Remove preexisting SBATCH lines except for --error, --output, and --job-name
            if [[ "$line" == "#SBATCH"* ]]; then
                if [[ "$line" == *"--error"* ]] || [[ "$line" == *"--output"* ]] || [[ "$line" == *"--job-name"* ]] || [[ "$line" == *"--array"* ]] || [[ "$line" == *"--cpus-per-gpu"* ]]; then
                    # Preserve these lines
                    echo "$line" >> "$temp_file"
                else
                    # Log and skip other SBATCH lines
                    echo "Removing SBATCH line: $line"
                    continue
                fi
            elif [[ "$line" == "export CHARLLM_FSDP="* ]] || [[ "$line" == "export CHARLLM_UNEVEN_PIPELINE="* ]] || [[ "$line" == "export HF_TOKEN="* ]]; then
                # Remove these lines
                echo "Removing line: $line"
                continue
            else
                # Add SBATCH directives before "module load anaconda3"
                if [[ "$line" == *"module load anaconda3"* ]]; then
                    # Add SBATCH directives only if environment variables are set
                    if [ -n "${CHARLLM_PARTITION}" ]; then
                        echo "#SBATCH --partition=${CHARLLM_PARTITION}" >> "$temp_file"
                    fi
                    if [ -n "${CHARLLM_QOS}" ]; then
                        echo "#SBATCH --qos=${CHARLLM_QOS}" >> "$temp_file"
                    fi
                    if [ -n "${CHARLLM_H200_NODELIST}" ]; then
                        echo "#SBATCH --nodelist=${CHARLLM_H200_NODELIST}" >> "$temp_file"
                    fi
                    if [ -n "${TIME_LIMIT}" ]; then
                        echo "#SBATCH --time=${TIME_LIMIT}" >> "$temp_file"
                    fi
                    if [ -n "${GPU_TYPE}" ]; then
                        echo "#SBATCH -C ${GPU_TYPE}" >> "$temp_file"
                    fi
                    if [ -n "${GPUS_PER_NODE}" ]; then
                        echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${MEM_PER_NODE}" ]; then
                        echo "#SBATCH --mem=${MEM_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${NUM_NODES}" ]; then
                        if [[ "$(basename "$file")" == *"download"* ]]; then
                            echo "#SBATCH --nodes=1" >> "$temp_file"
                        else
                            echo "#SBATCH --nodes=${NUM_NODES}" >> "$temp_file"
                        fi
                    fi
                    if [ -n "${NTASKS_PER_NODE}" ] && [[ "$(basename "$file")" != *"download"* ]]; then
                        echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${CPUS_PER_TASK}" ] && [[ "$(basename "$file")" != *"download"* ]]; then
                        echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> "$temp_file"
                    fi

                    # Add export statements
                    echo "export CHARLLM_FSDP=${FSDP_VALUE}" >> "$temp_file"
                    echo "export CHARLLM_UNEVEN_PIPELINE=${CHARLLM_UNEVEN_PIPELINE}" >> "$temp_file"
                    echo "export HF_TOKEN=${HF_TOKEN}" >> "$temp_file"
                fi

                # Write the original line to the temporary file
                echo "$line" >> "$temp_file"
            fi
        done < "$file"

        # Replace the original file with the modified version
        mv "$temp_file" "$file"
    fi
done

find "${CHARLLM_ROOT}/CharLLM-PPT/scripts/dataprep/" -mindepth 2 -type f \( -name "*.sh" -o -name "*.py" \) -print0 | while IFS= read -r -d '' file; do
    if grep -q "module load anaconda3" "$file"; then
        echo "Found 'module load anaconda3' in: $file"
        
        # Check if file path contains "-fsdp" to determine FSDP value
        if [[ "$file" == *"-fsdp"* ]]; then
            FSDP_VALUE=1
            echo "  Setting CHARLLM_FSDP=1 (detected -fsdp in path)"
        else
            FSDP_VALUE=${CHARLLM_FSDP}
            echo "  Setting CHARLLM_FSDP=${CHARLLM_FSDP} (default value)"
        fi
        
        # Create a temporary file with the modifications
        temp_file=$(mktemp)

        # Process the file line by line
        while IFS= read -r line; do
            # Remove preexisting SBATCH lines except for --error, --output, and --job-name
            if [[ "$line" == "#SBATCH"* ]]; then
                if [[ "$line" == *"--error"* ]] || [[ "$line" == *"--output"* ]] || [[ "$line" == *"--job-name"* ]] || [[ "$line" == *"--array"* ]] || [[ "$line" == *"--cpus-per-gpu"* ]]; then
                    # Preserve these lines
                    echo "$line" >> "$temp_file"
                else
                    # Log and skip other SBATCH lines
                    echo "Removing SBATCH line: $line"
                    continue
                fi
            elif [[ "$line" == "export CHARLLM_FSDP="* ]] || [[ "$line" == "export CHARLLM_UNEVEN_PIPELINE="* ]] || [[ "$line" == "export HF_TOKEN="* ]]; then
                # Remove these lines
                echo "Removing line: $line"
                continue
            else
                # Add SBATCH directives before "module load anaconda3"
                if [[ "$line" == *"module load anaconda3"* ]]; then
                    # Add SBATCH directives only if environment variables are set
                    if [ -n "${CHARLLM_PARTITION}" ]; then
                        echo "#SBATCH --partition=${CHARLLM_PARTITION}" >> "$temp_file"
                    fi
                    if [ -n "${CHARLLM_QOS}" ]; then
                        echo "#SBATCH --qos=${CHARLLM_QOS}" >> "$temp_file"
                    fi
                    if [ -n "${CHARLLM_H200_NODELIST}" ]; then
                        echo "#SBATCH --nodelist=${CHARLLM_H200_NODELIST}" >> "$temp_file"
                    fi
                    if [ -n "${TIME_LIMIT}" ]; then
                        echo "#SBATCH --time=${TIME_LIMIT}" >> "$temp_file"
                    fi
                    if [ -n "${GPU_TYPE}" ]; then
                        echo "#SBATCH -C ${GPU_TYPE}" >> "$temp_file"
                    fi
                    if [ -n "${GPUS_PER_NODE}" ]; then
                        echo "#SBATCH --gpus-per-node=${GPUS_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${MEM_PER_NODE}" ]; then
                        echo "#SBATCH --mem=${MEM_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${NUM_NODES}" ]; then
                        if [[ "$(basename "$file")" == *"download"* ]]; then
                            echo "#SBATCH --nodes=1" >> "$temp_file"
                        else
                            echo "#SBATCH --nodes=${NUM_NODES}" >> "$temp_file"
                        fi
                    fi
                    if [ -n "${NTASKS_PER_NODE}" ] && [[ "$(basename "$file")" != *"download"* ]]; then
                        echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}" >> "$temp_file"
                    fi
                    if [ -n "${CPUS_PER_TASK}" ] && [[ "$(basename "$file")" != *"download"* ]]; then
                        echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> "$temp_file"
                    fi

                    # Add export statements
                    echo "export CHARLLM_FSDP=${FSDP_VALUE}" >> "$temp_file"
                    echo "export CHARLLM_UNEVEN_PIPELINE=${CHARLLM_UNEVEN_PIPELINE}" >> "$temp_file"
                    echo "export HF_TOKEN=${HF_TOKEN}" >> "$temp_file"
                fi

                # Write the original line to the temporary file
                echo "$line" >> "$temp_file"
            fi
        done < "$file"

        # Replace the original file with the modified version
        mv "$temp_file" "$file"
    fi
done

echo "Step 2 completed."
echo ""
echo "Optimization sweep configuration complete!"

# Step 3: Check install.sbatch and only change ${CHARLLM_ROOT}
echo "Step 3: Updating CHARLLM_ROOT placeholder in install.sbatch only..."

INSTALL_DEFAULT="${CHARLLM_ROOT}/CharLLM-PPT/scripts/install.sbatch"
declare -a INSTALL_FILES=()

if [ -f "${INSTALL_DEFAULT}" ]; then
    INSTALL_FILES+=("${INSTALL_DEFAULT}")
else
    # Fallback: search for any install*.sbatch under scripts
    while IFS= read -r -d '' f; do
        INSTALL_FILES+=("$f")
    done < <(find "${CHARLLM_ROOT}/CharLLM-PPT/scripts" -type f -name "install*.sbatch" -print0)
fi

if [ "${#INSTALL_FILES[@]}" -eq 0 ]; then
    echo "No install.sbatch found under ${CHARLLM_ROOT}/CharLLM-PPT/scripts."
else
    for f in "${INSTALL_FILES[@]}"; do
        echo "Processing: $f"
        sed -i "s|\${CHARLLM_ROOT}|${CHARLLM_ROOT}|g" "$f"
    done
    echo "Step 3 completed."
fi