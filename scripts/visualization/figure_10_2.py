import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from vistools import *

def format_model_name(model: str) -> str:
    """Format model name with proper capitalization"""
    # Handle special cases first
    if model.startswith('gpt'):
        return model.replace('gpt3', 'GPT3').replace('b', 'B')
    elif model.startswith('llama'):
        return model.replace('llama3', 'Llama3').replace('b', 'B')
    elif model.startswith('mixtral'):
        return model.replace('mixtral', 'Mixtral').replace('b', 'B')
    return model

def get_model_parallel_configs() -> Dict[str, Dict[str, int]]:
    """Get mapping of models to their valid parallel configs and indices"""
    configs = {
        'gpt3-175b': {
            'TP2-PP16': 1, 
            'TP4-PP8': 2, 
            'TP8-PP4': 3,
            'TP1-PP32': 4
        },
        'llama3-70b': {
            'TP8-PP2': 1, 
            'TP4-PP4': 2, 
            'TP2-PP8': 3,
            'TP1-PP16': 4
        },
        'mixtral-8x7b': {
            'EP4-TP2-PP1': 1,
            'EP8-TP1-PP1': 2
        },
        'mixtral-8x22b': {
            'EP8-TP1-PP4': 1,
            'EP8-TP2-PP2': 2,
            'EP8-TP4-PP1': 3
        }
    }
    # Sort each model's configs by PP degree
    sorted_configs = {}
    for model, model_configs in configs.items():
        # Extract PP value and sort
        sorted_items = sorted(
            model_configs.items(),
            key=lambda x: int(re.search(r'PP(\d+)', x[0]).group(1))
        )
        sorted_configs[model] = dict(sorted_items)
    
    return sorted_configs
    
def get_pp_degree(config: str) -> int:
    """Extract PP degree from config name"""
    match = re.search(r'PP(\d+)', config)
    if match:
        return int(match.group(1))
    return 1  # Default to 1 if no PP found

CATEGORIES = {
    'Compute (MatMul)': '#4e79a7',    # Blue
    'SendRecv': '#59a14f',            # Green
    'AllGather': '#76b7b2',           # Teal
    'ReduceScatter': '#e15759',       # Red
    'AllReduce': '#f28e2b',           # Orange
    'Memory/Elementwise': '#edc948',   # Yellow
    'Layer Norm': '#b07aa1',          # Purple
    'Other': '#9c755f'                # Brown
}

def categorize_kernel(kernel_name: str) -> str:
    """Categorize kernel based on its name with specific comm types"""
    kernel_lower = kernel_name.lower()
    
    # Check for specific communication patterns
    if 'nccl' in kernel_lower or 'communication' in kernel_lower:
        if 'allreduce' in kernel_lower:
            return 'AllReduce'
        elif 'reducescatter' in kernel_lower:
            return 'ReduceScatter'
        elif 'allgather' in kernel_lower:
            return 'AllGather'
        elif 'sendrecv' in kernel_lower or 'broadcast' in kernel_lower:
            return 'SendRecv'
    
    # Check for buffer management and overlapping primitives
    if any(x in kernel_lower for x in ['pushsend', 'pushrecv', 'buffer', 'userbuffer']):
        return 'Memory/Elementwise'
    
    # Handle Triton fused kernels
    if 'triton_' in kernel_lower:
        if any(x in kernel_lower for x in ['sum', 'add', 'mul', 'div']):
            return 'Memory/Elementwise'
        if 'gemm' in kernel_lower:
            return 'Compute (MatMul)'
    
    # Check for compute kernels
    if any(x in kernel_lower for x in ['gemm', 'sdpa', 'matmul']):
        return 'Compute (MatMul)'
    elif 'elementwise' in kernel_lower or 'kernel' in kernel_lower:
        return 'Memory/Elementwise'
    elif 'norm' in kernel_lower:
        return 'Layer Norm'
    
    return 'Other'

def get_kernel_data(exp_path: str, config: str) -> Dict[str, float]:
    """Extract and categorize kernel execution times"""
    print(f"\nReading CSV file: {exp_path}")
    
    try:
        df = pd.read_csv(exp_path)
        duration_cols = [col for col in df.columns if 'total_duration' in col.lower()]
        
        # Get PP degree for normalization
        pp_degree = get_pp_degree(config)
        print(f"PP degree for {config}: {pp_degree}")
        
        # Initialize categories with zero values
        category_totals = {cat: 0.0 for cat in CATEGORIES.keys()}
        
        # Calculate total duration for each kernel and categorize
        for kernel in df['kernel_name'].unique():
            total_duration = df[df['kernel_name'] == kernel][duration_cols].sum().sum()
            normalized_duration = total_duration / (pp_degree * 1e6)  # Convert to seconds and normalize
            category = categorize_kernel(kernel)
            category_totals[category] += normalized_duration            
            
        return category_totals
        
    except Exception as e:
        print(f"Error processing {exp_path}: {e}")
        return None

def create_kernel_barchart(base_dir_1: str, base_dir_2: str, output_dir: str, crop_mode: str) -> None:
    """Create side-by-side stacked bar charts comparing kernel times between two configurations"""

    model_configs = get_model_parallel_configs()
        
    # Increase figure width to accommodate side-by-side bars
    figsize = (7, 1.8)  # Increased width for side-by-side bars
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set style and fonts
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 11,
    })
        
    def add_oom_text(ax, pos, width):
        """Helper function to add visible OOM markers"""
        ax.text(pos+0.5, 0.5, 'OOM', 
                ha='right', va='bottom',
                color='black', fontsize=8,
                rotation=90)
    
    current_pos = 0
    width = 1.6  
    spacing = 0.2  
    
    current_pos = 0
    bar_positions_1 = [] 
    bar_positions_2 = [] 
    bar_labels = []
    
    # Model positions calculation 
    model_positions = {}
    for model in model_configs:
        model_start = current_pos
        config_count = len(model_configs[model])
        # Account for both bars and spacing correctly
        model_width = config_count * (2 * width + spacing) + (config_count - 1)
        mid_point = model_start + model_width / 2
        model_positions[model] = (model_start, mid_point)
        # Add spacing between models
        current_pos += model_width + 2  
        
        if 'gpt' in model:
            mid_point -= 0.7
        elif 'llama' in model:
            mid_point -= 0.7
        elif 'mixtral-8x7b' in model:
            mid_point -= 0.7
        elif 'mixtral-8x22b' in model:
            mid_point -= 0.2
            
        model_positions[model] = (model_start, mid_point)
            

    current_pos = 0
    for model in model_configs:
        print(f"\nProcessing model: {model}")
        model_start = current_pos
        
        for config, idx in model_configs[model].items():
            print(f"  Config: {config}")
            
            # Update positions to include spacing
            bar_positions_1.append(current_pos)
            bar_positions_2.append(current_pos + width + spacing)
            config_label = config.replace('PP', 'P').replace('EP', 'E').replace('TP', 'T')
            bar_labels.append(config_label)
            
            # Process first directory
            csv_path_1 = os.path.join(base_dir_1, model, f"{model}_{idx}_00000_kernel_stats.csv")
            bottom_1 = 0
            if os.path.isfile(csv_path_1):
                kernel_data_1 = get_kernel_data(csv_path_1, config)
                if kernel_data_1:
                    for category in CATEGORIES:
                        duration = kernel_data_1[category]
                        if duration > 0:
                            ax.bar(current_pos, duration, width, bottom=bottom_1,
                                 color=CATEGORIES[category],
                                 label=category if current_pos == 0 else "",
                                 edgecolor='black', linewidth=0.5)
                            bottom_1 += duration
                else:
                    # Add OOM marker if kernel data processing failed
                    add_oom_text(ax, current_pos, width)
            else:
                # Add OOM marker if file doesn't exist
                add_oom_text(ax, current_pos, width)
                    
            # Process second directory
            csv_path_2 = os.path.join(base_dir_2, model, f"{model}_{idx}_01000_kernel_stats.csv")
            bottom_2 = 0
            if os.path.isfile(csv_path_2):
                kernel_data_2 = get_kernel_data(csv_path_2, config)
                if kernel_data_2:
                    for category in CATEGORIES:
                        duration = kernel_data_2[category]
                        if duration > 0:
                            ax.bar(current_pos + width + spacing, duration, width, bottom=bottom_2,
                                 color=CATEGORIES[category],
                                 label=category if current_pos == 0 else "",
                                 edgecolor='black', linewidth=0.5, alpha=1)
                            bottom_2 += duration
                else:
                    # Add OOM marker if kernel data processing failed
                    add_oom_text(ax, current_pos + width + spacing, width)
            else:
                # Add OOM marker if file doesn't exist
                add_oom_text(ax, current_pos + width + spacing, width)
            
            current_pos += (2 * width + spacing) + 1  # Move to next config pair with extra spacing
        
        # Add model separator
        if model != list(model_configs.keys())[0]:
            ax.axvline(x=model_start - (width + spacing*2) + 0.2, color='black', linestyle='-', alpha=0.7)
        
        current_pos += 1  # Space between models
    
    # Center labels between bar pairs
    ax.set_xticks([(pos1 + pos2)/2 for pos1, pos2 in zip(bar_positions_1, bar_positions_2)])
    ax.set_xticklabels(bar_labels, rotation=90, ha='center')

    ax.set_xlabel('Parallelism Configuration', fontsize=14, labelpad=10)
    ax.set_xlim(-2, current_pos - 1.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Create legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend = ax.legend(
        [by_label[cat] for cat in CATEGORIES if cat in by_label],
        [cat for cat in CATEGORIES if cat in by_label],
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.65),
        frameon=True,
        handlelength=0.8,
        handletextpad=0.4,
        columnspacing=0.6
    )
    
    ax.set_ylim(0, 60) # Set manually
    
    # Update model label placement code
    for model, (start, mid) in model_positions.items():
        ax.text(mid, ax.get_ylim()[1] * 1.03,
                format_model_name(model),
                ha='center', va='bottom', 
                fontsize=11)

    # Save plot with different cropping options (same as before)
    def save_with_consistent_size(output_file):
        plt.gcf().set_size_inches(figsize)
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        
    
    # Add y-axis label
    ax.set_ylabel('Time (seconds)', fontsize=14)
    ax.yaxis.set_label_coords(-0.09, 0.5)  # Adjust position
    
    
    # Save plot with different cropping options
    if crop_mode == 'full':
        # Save complete plot
        output_file = os.path.join(output_dir, f"model_parallel_kernel_times_{crop_mode}.pdf")
        save_with_consistent_size(output_file)
    
    elif crop_mode == 'no_legend':
        # Save without legend
        legend.remove()  # Use the legend object we created earlier
        output_file = os.path.join(output_dir, f"model_parallel_kernel_times_{crop_mode}.pdf")
        save_with_consistent_size(output_file)
    
    elif crop_mode == 'no_bottom':
        # Save without bottom labels
        ax.set_xlabel('')
        ax.set_xticklabels([])
        output_file = os.path.join(output_dir, f"model_parallel_kernel_times_{crop_mode}.pdf")
        save_with_consistent_size(output_file)
    
    elif crop_mode == 'data_only':
        # Save only the data area
        legend.remove()
        ax.set_xlabel('')
        ax.set_xticklabels([])
        output_file = os.path.join(output_dir, f"model_parallel_kernel_times_{crop_mode}.pdf")
        save_with_consistent_size(output_file)
    
    print(f"\nSaved parallel comparison plot to {output_file}")
    plt.close()
    

# Update main() function:
def main():
    base_dir_1 = "${CHARLLM_ROOT}/CharLLM-PPT/results/kineto-optimization-sweep-h200"
    base_dir_2 = "${CHARLLM_ROOT}/CharLLM-PPT/results/kineto-optimization-sweep-h200"
    output_dir = "${CHARLLM_ROOT}/CharLLM-PPT/figures/figures-2cols"
    
    # Create multiple versions with different crops
    for crop_mode in ['full', 'no_legend', 'no_bottom', 'data_only']:
        output_subdir = os.path.join(output_dir, crop_mode)
        os.makedirs(output_subdir, exist_ok=True)
        create_kernel_barchart(base_dir_1, base_dir_2, output_subdir, crop_mode)

    
if __name__ == "__main__":
    main()