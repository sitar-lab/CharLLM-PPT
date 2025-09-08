import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from matplotlib.gridspec import GridSpec
from vistools import *

    
def get_model_parallel_configs() -> Dict[str, Dict[str, int]]:
    """Get mapping of models to their valid parallel configs and indices"""
    # return {
    configs = {
        'gpt3-175b': {
            'TP2-PP16': 1, 
            'TP4-PP8': 2, 
            'TP8-PP4': 3,
            'TP1-PP32': 4,
            'TP8-FSDP': 5
        },
        'llama3-70b': {
            'TP8-PP2': 1, 
            'TP4-PP4': 2, 
            'TP2-PP8': 3,
            'TP1-PP16': 4,
            'TP8-FSDP': 5
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
                return -1
            m = re.search(r'PP(\d+)', x[0])
            return int(m.group(1)) if m else 999
        sorted_items = sorted(model_configs.items(), key=sort_key)
        sorted_configs[model] = dict(sorted_items)
    return sorted_configs
    
    
def format_model_name(model: str) -> str:
    """Format model name with proper capitalization"""
    # Handle special cases first
    if model.startswith('gpt'):
        return model.replace('gpt3', 'GPT3').replace('b', 'B')
    elif model.startswith('llama'):
        return model.replace('llama3', 'Llama3').replace('b', 'B')
    elif model.startswith('mixtral'):
        return model.replace('mixtral', 'Mixt').replace('b', 'B')
    return model

def create_stacked_plot(base_dir: str, output_dir: str):
    """Create vertically stacked plots for gpu_thermal, power, and utilization"""
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
    })
    
    model_configs = get_model_parallel_configs()
    
    # Collect data
    xlabels = []
    metric_data = {
        'gpu_thermal': {'avg': [], 'peak': [], 'intra_std': [], 'inter_std': []},
        'gpu_power': {'avg': [], 'peak': [], 'intra_std': [], 'inter_std': []},
        'gpu_clock': {'avg': [], 'peak': [], 'intra_std': [], 'inter_std': []},
    }
    
    # Collect execution metrics
    total_energies = []
    execution_times = []
    
    # Get data for each configuration
    for model in model_configs:
        for config, idx in model_configs[model].items():
            cleaned_config = config.replace('PP', 'P').replace('EP', 'E').replace('TP', 'T')
            
            xlabels.append(cleaned_config)

            # For FSDP, use a different base directory
            if 'FSDP' in config:
                base_exp_dir = f"{base_dir}-fsdp"
            else:
                base_exp_dir = base_dir
            
            exp_dir = os.path.join(base_exp_dir, f"{model}/{model}_{idx}_01000")
            
            # Get time and energy data
            exec_time = calculate_execution_time_from_profile(exp_dir)
            energy = calculate_total_energy(exp_dir)
            execution_times.append(exec_time if exec_time is not None else np.nan)
            total_energies.append(energy if energy is not None else np.nan)
            
            # Get metrics data
            for metric_name in metric_data.keys():
                collect_metric_data(exp_dir, metric_name, metric_data[metric_name])
    
    # Calculate efficiency
    execution_times = np.array(execution_times)
    total_energies = np.array(total_energies)
    with np.errstate(divide='ignore', invalid='ignore'):
        throughput = np.full_like(execution_times, np.nan, dtype=float)
        for i, label in enumerate(xlabels):
            batch_size = 128.0
            if not np.isnan(execution_times[i]) and execution_times[i] > 0:
                throughput[i] = batch_size / execution_times[i]
        energy_mj = total_energies / 1_000_000
        efficiency = throughput / energy_mj
    
    # Create figure with three vertically stacked subplots
    fig = plt.figure(figsize=(10, 6.6))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1], hspace=0.2)
    
    # Create axes for each metric with shared x-axis
    axes = {}
    metrics_order = ['gpu_thermal', 'gpu_power', 'gpu_clock']
    
    for idx, metric_name in enumerate(metrics_order):
        ax1 = fig.add_subplot(gs[idx])
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.0))
        axes[metric_name] = (ax1, ax2, ax3)
        
        if idx > 0:  # Share x-axis with first plot
            ax1.sharex(axes[metrics_order[0]][0])
    
    # Plot data and collect legend elements
    legend_lines = []
    legend_labels = []
    
    for metric_name in metrics_order:
        ax1, ax2, ax3 = axes[metric_name]
        plot_metric(ax1, ax2, ax3, metric_name, metric_data[metric_name], 
                   model_configs, xlabels, efficiency)
        
        # Only show x-labels on bottom plot
        if metric_name != 'gpu_clock':
            plt.setp(ax1.get_xticklabels(), visible=False)
        
        # Collect legend info from first plot
        if metric_name == 'gpu_thermal':
            legend_lines = ax1.get_lines()[:1]
            if metric_data[metric_name].get('peak', []):
                legend_lines.extend(ax1.get_lines()[1:2])
            legend_lines.extend(ax2.get_lines()[:2])
            legend_lines.extend(ax3.get_lines()[:1])
            legend_labels = [l.get_label() for l in legend_lines]
    
    # Add single legend at the top
    fig.legend(legend_lines, legend_labels,
              loc='upper center', 
              bbox_to_anchor=(0.5, 1.01),
              ncol=5,
              columnspacing=0.5,
              handletextpad=0.5,
              handlelength=1.5,
              frameon=True)
    
    plt.tight_layout(rect=[0, 0.0, 0.95, 0.95])
    
    # Save combined figure
    print(f"Saving final parallelism sweep figure to {output_dir}")
    plt.savefig(os.path.join(output_dir, "figure_4.pdf"),
                bbox_inches='tight',
                dpi=300)
    plt.close()

def add_oom_markers(ax, values):
    """Add OOM (Out of Memory) markers where values are NaN"""
    for i, val in enumerate(values):
        if np.isnan(val):
            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3)
            y_center = ax.get_ylim()[0] + 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        
            ax.text(i, y_center, "OOM", 
                   color='black', 
                   ha='center', 
                   va='center', 
                   fontsize=18, 
                   rotation=90)

def plot_metric(ax1, ax2, ax3, metric_name, data, model_configs, xlabels, efficiency):
    """Plot a single metric with its associated data"""
    
    # Track line segments for each model
    all_lines1, all_lines2, all_lines3 = [], [], []  # Removed lines3,4 for std
    
    # Plot model boundaries and data
    start_idx = 0
    for model in model_configs:
        n_configs = len(model_configs[model])
        end_idx = start_idx + n_configs
        
        if start_idx > 0:
            # Draw vertical line between models
            ax1.axvline(x=start_idx-0.5, color='black', linestyle='-', alpha=1.0)
        
        # Add model label only for the top subplot (gpu_thermal)
        if metric_name == 'gpu_thermal':
            mid_point = (start_idx + end_idx - 1) / 2
            ax1.text(mid_point, 1.04, format_model_name(model),
                    ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
                    fontsize=16)
        
        # Plot data for this model
        x_range = np.arange(start_idx, end_idx)
        
        # Plot average and peak on left y-axis
        line1 = ax1.plot(x_range, data['avg'][start_idx:end_idx], 
                       'b-', marker='o', label='Average' if start_idx == 0 else "",
                       linewidth=2.5)
        all_lines1.extend(line1)
        
        if len(data.get('peak', [])) > 0:
            line2 = ax1.plot(x_range, data['peak'][start_idx:end_idx],
                           'r-', marker='s', label='Peak' if start_idx == 0 else "",
                           linewidth=2.5)
            all_lines2.extend(line2)
        
        # Plot efficiency on rightmost y-axis
        line3 = ax3.plot(x_range, efficiency[start_idx:end_idx],
                       'k-', marker='x', label='Throughput/Energy' if start_idx == 0 else "",
                       linewidth=2, color='green')
        all_lines3.extend(line3)
        
        start_idx = end_idx
    
    # Format plot
    metric_info = metrics[metric_name]
    ax1.set_ylabel(metric_info['y1_label'])
    ax1.yaxis.set_label_coords(-0.09, 0.5)
    ax3.set_ylabel('Samples\n/(s·MJoule)', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Hide ax2 ticks and labels
    ax2.set_yticks([])
    ax2.set_yticklabels([])
    
    # Set x-axis labels and limits
    x = np.arange(len(xlabels))
    if metric_name == 'gpu_clock':
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels, ha='center', rotation=90)
        ax1.set_xlabel('Parallelism Configuration', fontsize=20, labelpad=10)  # Add x-axis label
    ax1.set_xlim(-0.5, len(xlabels)-0.5)

    # Color metric yticks/labels blue
    ax1.tick_params(axis='y', colors='blue')
    ax1.yaxis.label.set_color('blue')

    ax3.tick_params(axis='y', colors='green')
    ax3.yaxis.label.set_color('green')
    
    # Add OOM markers
    add_oom_markers(ax1, data['avg'])
    
def collect_metric_data(exp_dir, metric_name, data_dict):
    """Collect metric data from experiment directory"""
    if not os.path.exists(exp_dir):
        for key in data_dict:
            data_dict[key].append(np.nan)
        return
    
    values = []
    peak_values = []
    metric_info = metrics[metric_name]
    
    # Collect data from all GPUs
    for gpu_idx in range(32):
        avg_values = calculate_average_metric(exp_dir, metric_info['metric_type'], 
                                           gpu_idx, peak=False)
        if avg_values is not None:
            values.append(avg_values[metric_info['subtype']])
            if metric_info['peak']:
                peak_values.append(calculate_average_metric(exp_dir, 
                                metric_info['metric_type'],
                                gpu_idx, peak=True)[metric_info['subtype']])
    
    if values:
        values = np.array(values)
        if metric_info['metric_type'] == 'ib_bytes':
            node_values = values[:len(values)//8]
            values = np.repeat(node_values, 8)
        
        data_dict['avg'].append(np.mean(values))

        if metric_info['peak'] and peak_values:
            data_dict['peak'].append(np.max(peak_values))
        else:
            data_dict['peak'].append(np.nan)
    else:
        for key in data_dict:
            data_dict[key].append(np.nan)

def main():
    base_dir = "${CHARLLM_ROOT}/CharLLM-PPT/results/optimization-sweep-h200"
    output_dir = "${CHARLLM_ROOT}/CharLLM-PPT/figures"
    os.makedirs(output_dir, exist_ok=True)
    create_stacked_plot(base_dir, output_dir)

if __name__ == "__main__":
    main()