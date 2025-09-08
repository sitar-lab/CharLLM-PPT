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
            'TP8-FSDP': 5,
        },
        'mixtral-8x7b': {
            'EP4-TP2-PP1': 1,
            'EP8-TP1-PP1': 2,
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

def decode_bitmap_to_abbrev(bitmap):
    """Convert bitmap to abbreviated optimization names with enhanced handling"""
    opt_map = {
        '00000': 'base',
        '10000': 'cc',
        '01000': 'act',
        '11000': 'cc+act',
    }
    return opt_map.get(bitmap, 'other')

def get_config_sort_key(bitmap):
    """Enhanced sort key function for optimization types"""
    opt_order = {
        '00000': 0,  
        '10000': 1,
        '01000': 2,
        '11000': 3,
    }
    return opt_order.get(bitmap, 999)


def filter_opt_dirs(opt_dirs):
    """Helper function to filter invalid optimization combinations and check for OOM"""
    valid_dirs = []
    
    sync_exists = False
    for opt_dir in opt_dirs:
        bitmap = os.path.basename(opt_dir).split('_')[-1]
        is_sync = bitmap.endswith('1')
        if is_sync:
            sync_exists = True
            break
        
    if not sync_exists:
        return opt_dirs
    
    for opt_dir in opt_dirs:
        if not os.path.isdir(opt_dir):
            continue
            
        bitmap = os.path.basename(opt_dir).split('_')[-1]
        is_sync = bitmap.endswith('1')
        is_cc = bitmap.startswith('1')
        if (is_sync and not is_cc) or (not is_sync and is_cc):
            continue
            
        power_files = glob.glob(os.path.join(opt_dir, 'gpu_power_*.csv')) + \
                      glob.glob(os.path.join(opt_dir, 'power_*.csv'))
            
        if len(power_files) < 32:
            continue
        gpu_power_file = power_files[0]
        try:
            power_data = pd.read_csv(gpu_power_file, header=None)
            if len(power_data) < 10:
                print(f"Warning: Insufficient data points in {opt_dir}, likely OOM")
                continue
            exec_time = calculate_execution_time_from_profile(opt_dir)
            if exec_time is None or exec_time < 0:
                print(f"Warning: Invalid execution time in {opt_dir}, likely OOM")
                continue
            energy = calculate_total_energy(opt_dir)
            if energy is None or energy <= 0:
                print(f"Warning: Invalid energy data in {opt_dir}, likely OOM")
                continue
        except Exception as e:
            print(f"Warning: Error processing {opt_dir}: {e}, likely OOM")
            continue
        valid_dirs.append(opt_dir)
    if not valid_dirs:
        print("Warning: All directories filtered out, possible configuration issue")
    return valid_dirs


def normalize_by_model(values, model_configs, base_dir1, base_dir2):
    """Normalize values within each model (across all configs, using correct base_dir for FSDP/others)"""
    normalized = np.copy(values)
    current_idx = 0

    for model in model_configs:
        model_start_idx = current_idx
        model_points = 0

        # Count total points for this model
        for config, idx in model_configs[model].items():
            if 'FSDP' in config:
                base_exp_dir = os.path.join(base_dir1, model)
            else:
                base_exp_dir = os.path.join(base_dir2, model)
            pattern = f"{model}_{idx}_*"
            config_dirs = sorted(glob.glob(os.path.join(base_exp_dir, pattern)))
            config_dirs = [d for d in config_dirs if os.path.isdir(d)]
            model_points += len(config_dirs)

        if model_points == 0:
            continue

        model_slice = normalized[model_start_idx:model_start_idx + model_points]
        print(f"Normalizing {model}:")
        print(f"  slice range: {model_start_idx}:{model_start_idx + model_points}")
        print(f"  values: {model_slice}")

        valid_mask = ~np.isnan(model_slice)
        if np.any(valid_mask):
            model_max = np.max(model_slice[valid_mask])
            if model_max > 0:
                model_slice[valid_mask] = model_slice[valid_mask] / model_max
            normalized[model_start_idx:model_start_idx + model_points] = model_slice
            print(f"  max value: {model_max}")
            print(f"  normalized: {model_slice}")

        current_idx += model_points

    return normalized

def plot_disjoint_lines(ax, x, y, *args, **kwargs):
    """Plot lines only between consecutive valid (non-NaN) points, preserving all style kwargs and legend labels.
    Returns the Line2D object with the label (for legend), or None if no label is set.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    isnan = np.isnan(y)
    segments = []
    seg_x, seg_y = [], []
    for xi, yi, ni in zip(x, y, isnan):
        if not ni:
            seg_x.append(xi)
            seg_y.append(yi)
        else:
            if len(seg_x) > 1:
                segments.append((np.array(seg_x), np.array(seg_y)))
            seg_x, seg_y = [], []
    if len(seg_x) > 1:
        segments.append((np.array(seg_x), np.array(seg_y)))
    marker = kwargs.pop('marker', None)
    label = kwargs.pop('label', None)
    label_line = None
    for i, (seg_x, seg_y) in enumerate(segments):
        seg_kwargs = dict(kwargs)
        if i == 0 and label:
            seg_kwargs['label'] = label
        else:
            seg_kwargs['label'] = ""
        lines = ax.plot(seg_x, seg_y, *args, **seg_kwargs)
        # Save the line with the label for legend
        if i == 0 and label and lines:
            label_line = lines[0]
    # Always plot markers for all points (no label)
    marker_kwargs = dict(kwargs)
    if marker is not None:
        ax.plot(x[~isnan], y[~isnan], linestyle='None', marker=marker, **marker_kwargs)
    else:
        ax.plot(x[~isnan], y[~isnan], linestyle='None', **marker_kwargs)
    return label_line


def plot_metric(ax1, ax2, ax3, metric_name, data, model_configs, xlabels, efficiency, all_used_dirs, base_dir1, base_dir2):    
    """Plot a single metric with efficiency but without std dev"""
    # Track line segments for each model
    all_lines1, all_lines2, all_lines3 = [], [], []
    
    # Plot model boundaries and data
    start_idx = 0
    for model in model_configs:
        model_start_idx = start_idx
        model_dirs = 0
        last_config = None

        # First count total directories
        for config, idx in model_configs[model].items():
            if 'FSDP' in config:
                base_exp_dir = os.path.join(base_dir1, model)
            else:
                base_exp_dir = os.path.join(base_dir2, model)
            pattern = f"{model}_{idx}_*"
            opt_dirs = sorted(glob.glob(os.path.join(base_exp_dir, pattern)))
            opt_dirs = filter_opt_dirs(opt_dirs)
            model_dirs += len([d for d in opt_dirs if os.path.isdir(d)])

        # Now plot each config
        config_start = start_idx
        last_config_end = start_idx

        for config, idx in model_configs[model].items():
            if 'FSDP' in config:
                base_exp_dir = os.path.join(base_dir1, model)
            else:
                base_exp_dir = os.path.join(base_dir2, model)
            pattern = f"{model}_{idx}_*"
            opt_dirs = sorted(glob.glob(os.path.join(base_exp_dir, pattern)))
            opt_dirs = filter_opt_dirs(opt_dirs)
            n_dirs = len([d for d in opt_dirs if os.path.isdir(d)])
            if n_dirs == 0:
                continue

            config_end = config_start + n_dirs
            
            # Add separator between strategies
            if last_config is not None:
                ax1.axvline(x=config_start-0.5, color='black', linestyle='--', alpha=0.7)
            
            if metric_name == 'gpu_thermal':
                mid_point = (config_start + config_end - 1) / 2
                parallel_label = config.replace('PP', 'P').replace('EP', 'E').replace('TP', 'T')
                ax1.text(mid_point, 1.04, parallel_label,
                        ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
                        fontsize=16, color='black')
                                
            # For each config, use the correct slice of the data array
            data_slice = data['avg'][config_start:config_end]
            x_range = np.arange(config_start, config_end)
            # Only append the first line with a label
            line1 = plot_disjoint_lines(ax1, x_range, data_slice, 
                                    color='b', marker='o', linewidth=2.5, label='Average' if config_start == model_start_idx else "")
            if line1:
                all_lines1.append(line1)
            # Peak
            if len(data.get('peak', [])) > 0:
                peak_slice = data['peak'][config_start:config_end]
                line2 = plot_disjoint_lines(ax1, x_range, peak_slice, 
                                        color='r', marker='s', linewidth=2.5, label='Peak' if config_start == model_start_idx else "")
                if line2:
                    all_lines2.append(line2)
            # Efficiency
            eff_slice = efficiency[config_start:config_end]
            line3 = plot_disjoint_lines(ax3, x_range, eff_slice, 
                                    color='green', marker='x', linewidth=2, label='Efficiency (Samples/(s·MJoule))' if config_start == model_start_idx else "")
            if line3:
                all_lines3.append(line3)
            
            last_config_end = config_end
            config_start = config_end
            last_config = config
            
        # Add model separator and label
        if model_start_idx > 0:
            ax1.axvline(x=model_start_idx-0.5, color='black', linestyle='-', alpha=1.0)
            
        if metric_name == 'gpu_thermal':
            mid_point = (model_start_idx + last_config_end - 1) / 2
            ax1.text(mid_point, 1.29, format_model_name(model),
                    ha='center', va='bottom', transform=ax1.get_xaxis_transform(),
                    fontsize=16)
        
        start_idx = config_start
    
    # Format plot
    ax1.set_xlim(-0.5, len(xlabels)-0.5)
    
    # Return only lines with labels for legend
    legend_lines = []
    if len(all_lines1) > 0: 
        legend_lines.append(all_lines1[0])  # Average line
    if len(all_lines2) > 0: 
        legend_lines.append(all_lines2[0])  # Peak line
    if len(all_lines3) > 0:
        legend_lines.append(all_lines3[0])  # Efficiency line
    
    return legend_lines
    
def format_subplot(ax1, ax2, ax3, metric_info, xlabels, show_xlabels=False):
    """Format subplot axes and labels"""
    # Set labels
    ax1.set_ylabel(metric_info['y1_label'])
    ax1.yaxis.set_label_coords(-0.04, 0.5)
    
    # Update format_subplot() to reflect normalized efficiency label
    ax3.set_ylabel('Samples/\n(s*MJ)', fontsize=16)
    
    # Add grid
    ax1.grid(True, linestyle='-', alpha=0.3)
    
    # Set x-axis labels and limits
    x = np.arange(len(xlabels))
    if show_xlabels:
        ax1.set_xticks(x)
        ax1.set_xticklabels(xlabels, ha='center', rotation=90)
        ax1.set_xlabel('Optimization Configuration', fontsize=20, labelpad=10)
    else:
        ax1.set_xticks(x)
        ax1.set_xticklabels([])
    
    ax1.set_xlim(-0.5, len(xlabels)-0.5)
    
    # Color metric yticks/labels blue
    ax1.tick_params(axis='y', colors='blue')
    ax1.yaxis.label.set_color('blue')
    
    ax3.tick_params(axis='y', colors='green')
    ax3.yaxis.label.set_color('green')    

def add_oom_markers(ax, values, xlabels, model_configs):
    """Add OOM or N/A markers where values are NaN"""
    for i, val in enumerate(values):
        if np.isnan(val):

            ax.axvspan(i - 0.5, i + 0.5, color='lightgrey', alpha=0.3)
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


def create_sweep_plot(base_dir1, base_dir2, output_dir):
    """Create subplot for each metric showing model+config+optimization combinations"""
    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 16,
    })
    
    # Get configs and metrics to plot
    model_configs = get_model_parallel_configs()
        
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
    total_energies = []
    execution_times = []
    
    # Use lists for collecting time and energy data
    time_data = []
    energy_data = []
    
    # Create axes for each metric
    for idx, metric_name in enumerate(metrics_to_plot):
        ax1 = fig.add_subplot(gs[idx])
        ax2 = None  # Remove ax2
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.00))
        axes[metric_name] = (ax1, ax2, ax3)
    
    # First pass: Count total points
    total_points = 0
    for model in model_configs:
        for config, idx in model_configs[model].items():
            if 'FSDP' in config:
                base_exp_dir = os.path.join(base_dir1, model)
            else:
                base_exp_dir = os.path.join(base_dir2, model)
            pattern = f"{model}_{idx}_*"
            opt_dirs = sorted(glob.glob(os.path.join(base_exp_dir, pattern)))
            opt_dirs = filter_opt_dirs(opt_dirs)
            total_points += len([d for d in opt_dirs if os.path.isdir(d)])

    
    print(f"Total data points to collect: {total_points}")
    
    # Pre-allocate arrays
    for metric_name in metrics_to_plot:
        for key in metric_data[metric_name]:
            metric_data[metric_name][key] = np.full(total_points, np.nan)
    
    num_gpus = 32
    
    # Data collection
    data_index = 0
    for model in model_configs:
        print(f"\nProcessing model: {model}")
        for config, idx in model_configs[model].items():
            print(f"  Config: {config}")
            if 'FSDP' in config:
                base_exp_dir = os.path.join(base_dir1, model)
            else:
                base_exp_dir = os.path.join(base_dir2, model)
            pattern = f"{model}_{idx}_*"
            opt_dirs = sorted(glob.glob(os.path.join(base_exp_dir, pattern)))
            opt_dirs = sorted(opt_dirs, key=lambda x: get_config_sort_key(os.path.basename(x).split('_')[-1]))
            opt_dirs = filter_opt_dirs(opt_dirs)
            

            for opt_dir in opt_dirs:
                if not os.path.isdir(opt_dir):
                    continue
                # Get time and energy data
                exec_time = calculate_execution_time_from_profile(opt_dir)
                energy = calculate_total_energy(opt_dir)
                time_data.append(exec_time if exec_time is not None else np.nan)
                energy_data.append(energy if energy is not None else np.nan)
                
                # Update x-axis label
                bitmap = os.path.basename(opt_dir).split('_')[-1]
                if len(bitmap) == 5 and all(c in '01' for c in bitmap):
                    opt_abbrev = decode_bitmap_to_abbrev(bitmap)
                    config_label = opt_abbrev
                else:
                    config_label = 'base'
                xlabels.append(config_label)
                
                # Collect metrics data
                for metric_name in metrics_to_plot:
                    values = []
                    peak_values = []
                    metric_info = metrics[metric_name]
                    
                    for gpu_idx in range(num_gpus):
                        avg_values = calculate_average_metric(opt_dir, metric_info['metric_type'],
                                                           gpu_idx, peak=False)
                        if avg_values is not None:
                            values.append(avg_values[metric_info['subtype']])
                            if metric_info['peak']:
                                peak_values.append(calculate_average_metric(opt_dir,
                                                metric_info['metric_type'],
                                                gpu_idx, peak=True)[metric_info['subtype']])
                    
                    if values:
                        values = np.array(values)
                        metric_data[metric_name]['avg'][data_index] = np.mean(values)
                        if metric_info['peak'] and peak_values:
                            metric_data[metric_name]['peak'][data_index] = np.max(peak_values)
                
                data_index += 1

    all_used_dirs = []

    for opt_dir in opt_dirs:
        if not os.path.isdir(opt_dir):
            continue
        all_used_dirs.append(opt_dir)
    
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
        throughput = 128.0 / execution_times
        energy_mj = total_energies / 1_000_000
        raw_efficiency = throughput / energy_mj
        
        # Normalize efficiency within each model
        efficiency = normalize_by_model(raw_efficiency, model_configs, base_dir1, base_dir2)
    
    # Plot metrics
    for idx, metric_name in enumerate(metrics_to_plot):
        print(f"\nPlotting {metric_name}...")
        ax1, ax2, ax3 = axes[metric_name]
        metric_info = metrics[metric_name]
        
        lines = plot_metric(ax1, ax2, ax3, metric_name, metric_data[metric_name],
                            model_configs, xlabels, efficiency, all_used_dirs, base_dir1, base_dir2)
        
        if idx == 0:  # Store legend elements from first metric only
            legend_elements = [l for l in lines if l.get_label()]
        
        # Format subplot
        format_subplot(ax1, ax2, ax3, metric_info, xlabels, idx == len(metrics_to_plot)-1)
        
        # Add OOM markers
        add_oom_markers(ax1, metric_data[metric_name]['avg'], xlabels, model_configs)
        
    # Add single legend at the top with only unique labels
    if legend_elements:
        fig.legend(legend_elements,
                  [l.get_label() for l in legend_elements],
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 1.06),
                  ncol=5,
                  columnspacing=1.0,
                  handletextpad=0.5,
                  handlelength=1.5,
                  fontsize=16,
                  frameon=True)
    
    # Save plot
    plt.tight_layout(rect=[0, 0.0, 0.95, 0.95])
    output_file = os.path.join(output_dir, "figure_8.pdf")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved stacked plot to {output_file}")
    plt.close()

def main():
    base_dir1 = f"${CHARLLM_ROOT}/CharLLM-PPT/results/optimization-sweep-h200-fsdp"
    base_dir2 = f"${CHARLLM_ROOT}/CharLLM-PPT/results/optimization-sweep-h200"
    output_dir = f"${CHARLLM_ROOT}/CharLLM-PPT/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    create_sweep_plot(base_dir1, base_dir2, output_dir)

if __name__ == "__main__":
    main()