import glob
import os
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import ijson
import pandas as pd

def process_single_file(file_info):
    """Process a single JSON file and return its kernel statistics."""
    file_path, file_index, total_files = file_info
    
    file_size = os.path.getsize(file_path)
    print(
        f"Processing file {file_index} of {total_files}: {os.path.basename(file_path)}, "
        f"Size: {file_size / (1024 ** 3):.2f} GB"
    )
    
    # Dictionary to store kernel statistics
    kernel_stats = defaultdict(lambda: {"occurrences": 0, "total_duration": 0})
    
    # Parse and process JSON file
    with open(file_path, 'r') as f:
        for event in ijson.items(f, "traceEvents.item"):
            if event.get("cat") == "kernel":
                name = event.get("name")
                duration = event.get("dur", 0)
                kernel_stats[name]["occurrences"] += 1
                kernel_stats[name]["total_duration"] += duration
    
    return file_path, dict(kernel_stats)

def process_kineto_traces(model_name, base_dir):
    """Process Kineto traces for a given model and directory."""
    # Find all experiment directories for this model
    model_dir = os.path.join(base_dir, model_name)
    experiment_dirs = sorted(glob.glob(os.path.join(model_dir, "*")))
    
    print(f"Found {len(experiment_dirs)} experiment directories for {model_name}")
    
    all_results = {}
    
    # Create a process pool
    num_cores = 8
    pool = Pool(processes=num_cores)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        print(f"\nProcessing experiment: {exp_name}")
        
        # Find all kineto directories and select the highest numbered one
        kineto_dirs = glob.glob(f"{exp_dir}/*_kineto")
        if not kineto_dirs:
            print(f"No kineto directories found in {exp_dir}")
            continue
            
        # Sort by the number in the directory name and take the last one
        latest_kineto_dir = sorted(kineto_dirs, 
                                 key=lambda x: int(os.path.basename(x).split('_')[0]))[-1]
        
        # Find JSON files in the selected kineto directory
        trace_files = sorted(glob.glob(os.path.join(latest_kineto_dir, "*.json")))
        
        if not trace_files:
            print(f"No trace files found in {latest_kineto_dir}")
            continue
                    
        # Prepare file information for parallel processing
        file_infos = [(f, i+1, len(trace_files)) for i, f in enumerate(trace_files)]
        
        # Process files in parallel
        file_results = pool.map(process_single_file, file_infos)
        
        # Convert results to dictionary
        file_kernel_stats = dict(file_results)

        # Prepare data for the DataFrame
        all_kernels = set()
        for kernel_stats in file_kernel_stats.values():
            all_kernels.update(kernel_stats.keys())

        # Create a list of dictionaries for the DataFrame
        all_file_data = []
        for kernel_name in all_kernels:
            row = {"kernel_name": kernel_name}
            for file_path in trace_files:
                stats = file_kernel_stats.get(file_path, {}).get(
                    kernel_name, {"occurrences": 0, "total_duration": 0}
                )
                row[f"{file_path}_occurrences"] = stats["occurrences"]
                row[f"{file_path}_total_duration"] = stats["total_duration"]
                row[f"{file_path}_average_duration"] = (
                    stats["total_duration"] / stats["occurrences"] 
                    if stats["occurrences"] > 0 else 0
                )
            all_file_data.append(row)

        all_results[exp_name] = pd.DataFrame(all_file_data)
    
    # Close the process pool
    pool.close()
    pool.join()

    return all_results

def main():
    parser = argparse.ArgumentParser(description='Parse Kineto trace files')
    parser.add_argument('--model', type=str, default="gpt3-175b",
                       help='Model name (default: gpt3-175b)')
    args = parser.parse_args()

    # Define paths
    base_dir = "${CHARLLM_ROOT}/CharLLM-PPT/results/optimization-sweep-h200-fsdp"
    output_dir = "${CHARLLM_ROOT}/CharLLM-PPT/results/kineto-optimization-sweep-h200-fsdp"
    os.makedirs(output_dir, exist_ok=True)

    # Process trace files for each experiment
    results_dict = process_kineto_traces(args.model, base_dir)
    
    if results_dict:
        model_dir = os.path.join(output_dir, args.model)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save each experiment's results to a separate CSV file
        for exp_name, df in results_dict.items():
            output_csv = os.path.join(model_dir, f"{exp_name}_kernel_stats.csv")
            df.to_csv(output_csv, index=False)
            print(f"\nSaved kernel statistics for {exp_name} to {output_csv}")
            
            # Calculate and display summary statistics
            print(f"\nKernel Statistics Summary for {exp_name}:")
            print(f"Total unique kernels: {len(df)}")

if __name__ == "__main__":
    main()