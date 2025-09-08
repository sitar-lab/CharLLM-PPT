import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from matplotlib.gridspec import GridSpec
from vistools import *

    
def get_model_parallel_configs() -> Dict[str, Dict[str, int]]:
    """Get mapping of models to their valid parallel configs and indices"""
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
                return -1  # FSDP first
            m = re.search(r'PP(\d+)', x[0])
            return int(m.group(1)) if m else 999
        sorted_items = sorted(model_configs.items(), key=sort_key)
        sorted_configs[model] = dict(sorted_items)
    return sorted_configs
    
def decode_config_label(config_name: str, mb_size: int) -> str:
    """Convert config name to display label with microbatch size"""
    return str(mb_size)  # Just return the number

def normalize_by_model(values, model_configs, microbatch_sizes):
    """Normalize values so that the max for each model is 1 (across all configs and microbatch sizes)."""
    normalized = np.copy(values)
    current_idx = 0
    for model in model_configs:
        n_points = len(model_configs[model]) * len(microbatch_sizes)
        model_slice = normalized[current_idx:current_idx + n_points]
        valid_mask = ~np.isnan(model_slice)
        if np.any(valid_mask):
            model_max = np.max(model_slice[valid_mask])
            if model_max > 0:
                model_slice[valid_mask] = model_slice[valid_mask] / model_max
            normalized[current_idx:current_idx + n_points] = model_slice
        current_idx += n_points
    return normalized

def plot_metric(ax1, ax2, ax3, metric_name, data, model_configs, xlabels, efficiency, base_dir):
    """Plot a single metric with its associated data"""
    # Track line segments for each model
    all_lines1, all_lines2, all_lines3 = [], [], []
    
    # Plot model boundaries and data
    start_idx = 0
    for model in model_configs:
        model_start_idx = start_idx
        model_dirs = 0
        last_config = None
        
        # First count total directories for this model
        for config, idx in model_configs[model].items():
            pattern = f"{model}_{idx}_*"
            opt_dirs_mp = sorted(glob.glob(os.path.join(base_dir, model, pattern)))
            opt_dirs_fsdp = sorted(glob.glob(os.path.join(f"{base_dir}-fsdp", model, pattern)))
            opt_dirs =  opt_dirs_fsdp + opt_dirs_mp

            model_dirs += len([d for d in opt_dirs if os.path.isdir(d)])
        
        # Now plot each config
        config_start = start_idx
        last_config_end = start_idx  # Track end of last config for label positioning
        
        for config, idx in model_configs[model].items():
            pattern = f"{model}_{idx}_*"
            
            opt_dirs_mp = sorted(glob.glob(os.path.join(base_dir, model, pattern)))
            opt_dirs_fsdp = sorted(glob.glob(os.path.join(f"{base_dir}-fsdp", model, pattern)))
            opt_dirs =  opt_dirs_fsdp + opt_dirs_mp

            n_dirs = len([d for d in opt_dirs if os.path.isdir(d)])
            if n_dirs == 0:
                continue
                
            # Calculate indices for this config
            config_end = config_start + n_dirs
            
            # Add vertical separator between different parallel strategies
            if last_config is not None:
                ax1.axvline(x=config_start-0.5, color='black', linestyle='--', alpha=0.7)
        
            if metric_name == 'gpu_thermal':
                mid_point = (config_start + config_end - 1) / 2
                parallel_label = config.replace('PP', 'P').replace('EP', 'E').replace('TP', 'T')
                ax1.text(mid_point, 1.04, parallel_label,
                        ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
                        fontsize=16, color='black')
            
            # Plot config data
            x_range = np.arange(config_start, config_end)
            data_slice = data['avg'][config_start:config_end]
            
            if len(x_range) != len(data_slice):
                print(f"Warning: Skipping plot due to dimension mismatch ({len(x_range)} vs {len(data_slice)})")
                continue
            
            # Plot average
            line1 = ax1.plot(x_range, data_slice, 
                           'b-', marker='o', label='Average' if config_start == model_start_idx else "",
                           linewidth=2)
            all_lines1.extend(line1)
            
            # Plot peak if available
            if len(data.get('peak', [])) > 0:
                peak_slice = data['peak'][config_start:config_end]
                if len(x_range) == len(peak_slice):
                    line2 = ax1.plot(x_range, peak_slice,
                                   'r-', marker='s', label='Peak' if config_start == model_start_idx else "",
                                   linewidth=2)
                    all_lines2.extend(line2)
            
            # Plot efficiency
            eff_slice = efficiency[config_start:config_end]
            if len(x_range) == len(eff_slice):
                line3 = ax3.plot(x_range, eff_slice,
                               'k-', marker='x', label='Throughput/Energy' if config_start == model_start_idx else "",
                               linewidth=2, color='green')
                all_lines3.extend(line3)
            
            last_config_end = config_end
            config_start = config_end
            last_config = config
            
        # Add model separator and label after all configs
        if model_start_idx > 0:
            ax1.axvline(x=model_start_idx-0.5, color='black', linestyle='-', alpha=0.7)
            
        if metric_name == 'gpu_thermal':
            mid_point = (model_start_idx + last_config_end - 1) / 2
            ax1.text(mid_point, 1.25, format_model_name(model),
                    ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
                    fontsize=16)
        
        start_idx = config_start
        
    # Format plot
    ax1.set_xlim(-0.5, len(xlabels)-0.5)
    
    # Return only lines with labels for legend
    legend_lines = []
    if len(all_lines1) > 0: legend_lines.append(all_lines1[0])
    if len(all_lines2) > 0: legend_lines.append(all_lines2[0])
    if len(all_lines3) > 0: legend_lines.append(all_lines3[0])
    
    return legend_lines

def format_subplot(ax1, ax2, ax3, metric_info, xlabels, show_xlabels=True):
    """Format subplot axes and labels"""
    # Set labels
    ax1.set_ylabel(metric_info['y1_label'])
    ax1.yaxis.set_label_coords(-0.04, 0.5)
    # Remove ax2 ylabel and label coordinates
    ax3.set_ylabel('Norm.\nEfficiency', fontsize=18)
    
    # Add grid
    ax1.grid(True, linestyle='-', alpha=0.3)
    
    # Set log scale for NVLink and PCIe bytes with better formatting
    if metric_info['metric_type'] in ['nvlink_bytes', 'pcie_bytes']:
        ax1.set_yscale('log', base=2)
        
        # Get current y-axis limits
        ymin, ymax = ax1.get_ylim()
        
        # Create more frequent tick positions as powers of 2
        min_pow = int(np.ceil(np.log2(ymin)))
        max_pow = int(np.ceil(np.log2(ymax)))
        major_ticks = [2**i for i in range(min_pow, max_pow + 1, 3)]

        ax1.set_yticks(major_ticks)
        
        # Format major tick labels
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:g}'.format(y)))
        
    # Set x-axis labels and limits
    x = np.arange(len(xlabels))
    if show_xlabels:
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels, ha='center', rotation=0)  # Changed rotation from 90 to 45
        ax1.set_xlabel('Microbatch Size', fontsize=20, labelpad=10)  # Add x-axis label
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
    
    ax1.set_xlim(-0.5, len(xlabels)-0.5)
    
    # Color metric yticks/labels blue
    ax1.tick_params(axis='y', colors='blue')
    ax1.yaxis.label.set_color('blue')
    
    ax3.tick_params(axis='y', colors='green')
    ax3.yaxis.label.set_color('green')
    
def add_oom_markers(ax, values):
    """Add OOM (Out of Memory) markers where values are NaN"""
    for i, val in enumerate(values):
        if np.isnan(val):
            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3)
            
            # Different y-center calculation based on scale
            if ax.get_yscale() == 'log':
                # For log scale, use geometric mean of limits
                ymin, ymax = ax.get_ylim()
                y_center = np.sqrt(ymin * ymax)
            else:
                # For linear scale, use arithmetic mean
                y_center = ax.get_ylim()[0] + 0.5 * (ax.get_ylim()[1] - ax.get_ylim()[0])
            
            ax.text(i, y_center, "OOM", 
                   color='black', 
                   ha='center', 
                   va='center', 
                   fontsize=14, 
                   rotation=90)
            
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

def create_sweep_plot(base_dir: str, output_dir: str):
    """Create subplot for each metric showing model+config+optimization combinations"""
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
    })
    
    # Get configs and metrics to plot
    model_configs = get_model_parallel_configs()
    num_gpus = 32
    
    microbatch_sizes = [1, 2, 4, 8]
    print(f"Found {len(model_configs)} models to process")
    
    # Create figure with three vertically stacked subplots
    fig = plt.figure(figsize=(24, 7))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.2)
    axes = {}
    
    # Initialize data structures
    data_index = 0
    xlabels = []
    metrics_to_plot = ['gpu_thermal', 'gpu_power', 'gpu_clock']
    metric_data = {name: {'avg': [], 'peak': [], 'std': [], 'intra_std': [], 'inter_std': []} 
                  for name in metrics_to_plot}
    
    # Use lists for collecting time and energy data
    time_data = []
    energy_data = []
    
    # Create axes for each metric
    for idx, metric_name in enumerate(metrics_to_plot):
        ax1 = fig.add_subplot(gs[idx])
        ax2 = None
        ax3 = ax1.twinx() 
        ax3.spines["right"].set_position(("axes", 1.0))
        axes[metric_name] = (ax1, None, ax3)  # Pass None for ax2
        
    # First pass: Count total points
    total_points = 0
    for model in model_configs:
        for config, idx in model_configs[model].items():
            for mb_size in microbatch_sizes:
                # If fsdp is in config, append base_dir with -fsdp
                base_exp_dir = f"{base_dir}-fsdp" if 'FSDP' in config else base_dir

                base_exp_dir = os.path.join(base_exp_dir, model)
                 
                pattern = f"{model}_{idx}_mbs{mb_size}"
                if os.path.exists(os.path.join(base_exp_dir, pattern)):
                    total_points += 1
    
    print(f"Total data points to collect: {total_points}")
    
    # Pre-allocate arrays
    for metric_name in metrics_to_plot:
        for key in metric_data[metric_name]:
            metric_data[metric_name][key] = np.full(total_points, np.nan)
    
    # Collect data
    data_index = 0
    for model in model_configs:
        print(f"\nProcessing model: {model}")
        for config, idx in model_configs[model].items():
            print(f"  Config: {config}")
            
            for mb_size in microbatch_sizes:
                # If fsdp is in config, append base_dir with -fsdp
                base_exp_dir = f"{base_dir}-fsdp" if 'FSDP' in config else base_dir
                base_exp_dir = os.path.join(base_exp_dir, model)                
                exp_dir = os.path.join(base_exp_dir, f"{model}_{idx}_mbs{mb_size}")
                
                if not os.path.isdir(exp_dir):
                    continue
                
                # Get time and energy data
                exec_time = calculate_execution_time_from_profile(exp_dir)
                energy = calculate_total_energy(exp_dir)
                time_data.append(exec_time if exec_time is not None else np.nan)
                energy_data.append(energy if energy is not None else np.nan)
                
                # Update x-axis label with microbatch size
                config_label = decode_config_label(config, mb_size)
                xlabels.append(config_label)
                
                # Collect metrics data
                for metric_name in metrics_to_plot:
                    values = []
                    peak_values = []
                    metric_info = metrics[metric_name]
                    
                    for gpu_idx in range(num_gpus):
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
                        metric_data[metric_name]['avg'][data_index] = np.mean(values)
                        if metric_info['peak'] and peak_values:
                            metric_data[metric_name]['peak'][data_index] = np.max(peak_values)
                
                data_index += 1
    
    # Print data collection summary
    print("\nData Collection Summary:")
    for metric_name in metrics_to_plot:
        valid_count = np.sum(~np.isnan(metric_data[metric_name]['avg']))
        total_count = len(metric_data[metric_name]['avg'])
        print(f"{metric_name}: {valid_count}/{total_count} valid data points")
    
    # Calculate efficiency using collected data
    execution_times = np.array(time_data)
    total_energies = np.array(energy_data)
    with np.errstate(divide='ignore', invalid='ignore'):
        throughput = np.full_like(execution_times, np.nan, dtype=float)
        data_index = 0
        for model in model_configs:
            for config, idx in model_configs[model].items():
                for mb_size in microbatch_sizes:
                    # Only compute if this data point exists
                    if data_index >= len(execution_times):
                        continue
                    batch_size = 128.0
                    if not np.isnan(execution_times[data_index]) and execution_times[data_index] > 0:
                        throughput[data_index] = batch_size / execution_times[data_index]
                    data_index += 1
        energy_mj = total_energies / 1_000_000
        efficiency = throughput / energy_mj
        # Normalize efficiency within each model
        efficiency = normalize_by_model(efficiency, model_configs, microbatch_sizes)        
    
    # Plot metrics
    for idx, metric_name in enumerate(metrics_to_plot):
        print(f"\nPlotting {metric_name}...")
        ax1, ax2, ax3 = axes[metric_name]
        metric_info = metrics[metric_name]
        
        lines = plot_metric(ax1, ax2, ax3, metric_name, metric_data[metric_name],
                          model_configs, xlabels, efficiency, base_dir)

        if idx == 0:  # Store legend elements from first metric only
            legend_elements = [l for l in lines if l.get_label()]
        
        # Format subplot
        format_subplot(ax1, ax2, ax3, metric_info, xlabels, idx == len(metrics_to_plot)-1)
        
        # Add OOM markers
        add_oom_markers(ax1, metric_data[metric_name]['avg'])
    
    # Add single legend at the top with only unique labels
    if legend_elements:
        fig.legend(legend_elements,
                  [l.get_label() for l in legend_elements],
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 1.04),
                  ncol=5,
                  columnspacing=1.0,
                  handletextpad=0.5,
                  handlelength=1.5,
                  fontsize=16,
                  frameon=True)
    
    # Save plot
    plt.tight_layout(rect=[0, 0.0, 0.95, 0.95])
    output_file = os.path.join(output_dir, "figure_13.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked plot to {output_file}")
    plt.close()

def main():
    base_dir = f"${CHARLLM_ROOT}/CharLLM-PPT/results/microbatch-sweep-h200"
    output_dir = f"${CHARLLM_ROOT}/CharLLM-PPT/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    create_sweep_plot(base_dir, output_dir)

if __name__ == "__main__":
    main()