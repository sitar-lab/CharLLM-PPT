import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
from vistools import *

def get_model_parallel_configs() -> Dict[str, Dict[str, int]]:
    """Get mapping of models to their valid parallel configs and indices"""
    configs = {
        'gpt3-175b': {
            'TP2-PP16': 1, 
            'TP4-PP8': 2, 
            'TP8-PP4': 3,
            'TP1-PP32': 4,
        },
        'llama3-70b': {
            'TP8-PP2': 1, 
            'TP4-PP4': 2, 
            'TP2-PP8': 3,
            'TP1-PP16': 4,
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
        def sort_key(x):
            if 'FSDP' in x[0]:
                return -1  # FSDP first
            m = re.search(r'PP(\d+)', x[0])
            return int(m.group(1)) if m else 999
        sorted_items = sorted(model_configs.items(), key=sort_key)
        sorted_configs[model] = dict(sorted_items)
    return sorted_configs

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

def get_kernel_data(exp_path: str, config: str) -> Dict[str, List[float]]:
    """Extract and categorize kernel execution times for each rank"""
    print(f"\nReading CSV file: {exp_path}")
    
    try:
        df = pd.read_csv(exp_path)
        # Get individual rank durations (even indices only, since the durations are total, avg, total, avg, ...)
        rank_cols = [col for col in df.columns if 'duration' in col.lower()]
        even_rank_cols = rank_cols[::2]  # Take every other column
        
        # Initialize categories with zero values for each rank
        category_totals = {cat: [0.0] * len(even_rank_cols) for cat in CATEGORIES.keys()}
        
        # Calculate duration for each kernel and rank
        for kernel in df['kernel_name'].unique():
            kernel_data = df[df['kernel_name'] == kernel][even_rank_cols].values[0]
            category = categorize_kernel(kernel)
            for rank, duration in enumerate(kernel_data):
                category_totals[category][rank] += duration / 1e6  # Convert to seconds
            
        return category_totals
        
    except Exception as e:
        print(f"Error processing {exp_path}: {e}")
        return None

def create_rank_plots(base_dir: str, output_dir: str, target_model: str, crop_mode: str = 'full'):
    """Create single plot with configs across x-axis and ranks as grouped bars"""
    model_configs = get_model_parallel_configs()[target_model]
    
    # First pass to determine total width needed and max time
    bar_width = 0.6  # Reduced bar width to add spacing between bars
    group_spacing = 2.0    # Space between config groups
    total_width = 0
    max_ranks = 0
    global_max_time = 0  # Track global maximum time
    
    for config, config_idx in model_configs.items():
        csv_path = os.path.join(base_dir, target_model, f"{target_model}_{config_idx}_mbs4_kernel_stats.csv")
        if os.path.isfile(csv_path):
            kernel_data = get_kernel_data(csv_path, config)
            if kernel_data:
                n_ranks = len(list(kernel_data.values())[0])
                max_ranks = max(max_ranks, n_ranks)
                total_width += n_ranks * bar_width + group_spacing
                # Calculate max time for this config
                for rank in range(n_ranks):
                    rank_total = sum(kernel_data[category][rank] for category in CATEGORIES)
                    global_max_time = max(global_max_time, rank_total)
    
    # Create figure with calculated width
    fig, ax = plt.subplots(figsize=(7, 2.8))
    
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
    
    current_x = 1.0  # Start with some margin
    max_time = 0
    all_xticks = []
    all_xticklabels = []
    
    # Process each config
    for config, config_idx in model_configs.items():
        csv_path = os.path.join(base_dir, target_model, f"{target_model}_{config_idx}_mbs4_kernel_stats.csv")
        if os.path.isfile(csv_path):
            kernel_data = get_kernel_data(csv_path, config)
            if kernel_data:
                n_ranks = len(list(kernel_data.values())[0])
                
                rank_positions = []
                
                for rank in range(n_ranks):
                    rank_x = current_x + rank * (bar_width + 0.2)  # Add 0.2 spacing between bars
                    rank_positions.append(rank_x)
                    bottom = 0
                    
                    for category in CATEGORIES:
                        duration = kernel_data[category][rank]
                        if duration > 0:
                            ax.bar(rank_x, duration, bar_width,
                                 bottom=bottom,
                                 color=CATEGORIES[category],
                                 label=category if current_x == 1.0 and rank == 0 else "",
                                 edgecolor='black', linewidth=0.5)
                            bottom += duration
                    
                    max_time = max(max_time, bottom)
                
                global_max_time = 40 # Set manually
                ax.set_ylim(0, global_max_time)
                # Add config label using global max time
                config_center = (rank_positions[0] + rank_positions[-1]) / 2
                ax.text(config_center, global_max_time * 1.04, f"{config}",
                       ha='center', va='bottom', fontsize=10)
                
                # Store tick positions and labels for this config
                tick_positions = [rank_positions[i] for i in range(0, n_ranks, 4)]
                tick_labels = [f'{i}' for i in range(0, n_ranks, 4)]
                all_xticks.extend(tick_positions)
                all_xticklabels.extend(tick_labels)
                
                # Add vertical separator
                if current_x > 1.0:
                    ax.axvline(x=current_x - group_spacing*0.7, color='black', linestyle='-', alpha=0.5)
                
                current_x += n_ranks * (bar_width + 0.2) + group_spacing
    
    # Set all x-ticks at once after plotting
    ax.set_xticks(all_xticks)
    ax.set_xticklabels(all_xticklabels, rotation=0, ha='center', fontsize=10)
    
    ax.yaxis.set_label_coords(-0.07, 0.5)  # Adjust label position
    ax.set_xlabel('PP_Rank', fontsize=14)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_xlim(0, current_x - group_spacing + 1)  # Add margin at the end
    
    # Set y-ticks fontsize
    ax.tick_params(axis='y', labelsize=10)
    
    # Add "ReduceScatter" item to legend
    ax.bar(0, 0, color=CATEGORIES['ReduceScatter'], label='ReduceScatter',
             edgecolor='black', linewidth=0.5)
    
    # In the legend, switch "ReduceScatter" and "Other" positions
    legend_order = ['Compute (MatMul)', 'SendRecv', 'AllGather', 'AllReduce',
                    'Memory/Elementwise', 'Layer Norm', 'ReduceScatter', 'Overlap', 'Other']
    handles, labels = ax.get_legend_handles_labels()
    ordered_handles = [handles[labels.index(cat)] for cat in legend_order if cat in labels]
    ordered_labels = [cat for cat in legend_order if cat in labels]
    
    # Add legend with more space at top
    ax.legend(ordered_handles, ordered_labels, ncol=4, bbox_to_anchor=(0.5, 1.55), loc='upper center',
             frameon=True, handlelength=0.8, handletextpad=0.4, columnspacing=0.6)
    
    # Adjust layout with more top margin
    plt.tight_layout()
    plt.subplots_adjust(top=0.8)  # Increase top margin for legend
    
    # Save plot with different cropping options
    if crop_mode == 'full':
        # Save complete plot
        output_file = os.path.join(output_dir, f"{target_model}_rank_comparison_{crop_mode}.pdf")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    elif crop_mode == 'no_legend':
        # Save without legend
        plt.legend().remove()  # Remove legend
        # Reapply font sizes
        plt.rcParams.update({
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
        output_file = os.path.join(output_dir, f"{target_model}_rank_comparison_{crop_mode}.pdf")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    elif crop_mode == 'no_bottom':
        # Save without bottom labels and title
        ax.set_xlabel('')  # Remove x-axis label
        # Reapply font sizes
        plt.rcParams.update({
            'ytick.labelsize': 14,
        })
        output_file = os.path.join(output_dir, f"{target_model}_rank_comparison_{crop_mode}.pdf")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    elif crop_mode == 'data_only':
        # Save only the data area
        plt.legend().remove()  # Remove legend
        ax.set_xlabel('')  # Remove x-axis label
        ax.set_ylabel('')  # Remove y-axis label
        ax.set_xticklabels([])  # Remove x-axis tick labels
        ax.set_yticklabels([])  # Remove y-axis tick labels
        # Reapply font sizes
        plt.rcParams.update({
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
        })
        output_file = os.path.join(output_dir, f"{target_model}_rank_comparison_{crop_mode}.pdf")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
    
    print(f"\nSaved rank comparison plot to {output_file}")
    plt.close()
    
def main():
    base_dir = "${CHARLLM_ROOT}/CharLLM-PPT/results/kineto-microbatch-sweep-h200/"
    output_dir = "${CHARLLM_ROOT}/CharLLM-PPT/figures/microbatch-sweep-h200-mbs4/"
    
    # Select model to analyze
    target_model = 'gpt3-175b'  # Change this to analyze different models
    
    os.makedirs(output_dir, exist_ok=True)
    # Create multiple versions with different crops
    for crop_mode in ['full', 'no_legend', 'no_bottom', 'data_only']:
        output_subdir = os.path.join(output_dir, crop_mode)
        os.makedirs(output_subdir, exist_ok=True)
        create_rank_plots(base_dir, output_subdir, target_model, crop_mode)

if __name__ == "__main__":  
    main()