import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm 
from pathlib import Path
from vistools import *

def get_parallel_strategy(model_name, parallel_idx):
    strategies = {
        'gpt3-175b': {
            '5': 'TP8-FSDP'
        },
        'llama3-70b': {
            '5': 'TP8-FSDP'
        },
    }
    
    return strategies.get(model_name, {}).get(str(parallel_idx), f'unknown_{parallel_idx}')

def format_model_name(model: str) -> str:
    """Format model name with shorter, cleaner labels"""
    # Handle special cases first
    if model.startswith('gpt'):
        return model.replace('gpt3-175b', 'G175').replace('gpt3-70b', 'G70')
    elif model.startswith('llama'):
        return model.replace('llama3-70b', 'L70')
    elif model.startswith('mixtral'):
        return (model.replace('mixtral-8x22b', 'M8x22')
                    .replace('mixtral-8x7b', 'M8x7'))
    return model

def format_strategy_name(strategy: str) -> str:
    """Format parallel strategy with shorter labels"""
    # Replace common patterns with shorter versions
    return (strategy.replace('TP', 'T')
                   .replace('PP', 'P')
                   .replace('EP', 'E')
                   .replace('-', ''))

def create_heatmap(data, title, output_path, cmap='YlOrRd', fmt='.1f', execution_times=None, peak=False):
    """Create and save a compact heatmap suitable for paper column format"""
    # Determine if we should show the average row based on metric type
    metric_type = title.split(' - ')[1].split(' (')[0].lower()
    show_avg = not (metric_type.startswith('nvlink') or metric_type.startswith('pcie'))

    # Calculate means and figure size
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    num_rows = len(data.index)
    num_cols = len(data.columns)
    fig_width = 7
    
    if show_avg:
        plt.figure(figsize=(fig_width, 2.3))
    else:
        # plt.figure(figsize=(fig_width, 2.1))
        plt.figure(figsize=(fig_width, 0.6))
    
    # Update GridSpec based on whether we show average row
    if show_avg:
        gs = plt.GridSpec(2, 2, width_ratios=[num_cols, 1],
                         height_ratios=[1, num_rows],
                         hspace=0.02, wspace=0.05)
    else:
        gs = plt.GridSpec(1, 2, width_ratios=[num_cols, 1],
                         hspace=0.02, wspace=0.05)
    
    # Create the main heatmap with a normalized colorscale
    vmin = min(data.min().min(), col_means.min(), row_means.min())
    vmax = max(data.max().max(), col_means.max(), row_means.max())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Set smaller font sizes for paper format
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 11,
    })
    
    # Use log normalization for networking metrics
    if any(metric in title.lower() for metric in ['nvlink', 'pcie', 'ib']):
        vmin = max(2, vmin)  # Set minimum to 2^1 = 2
        vmin = 2 ** np.floor(np.log2(vmin))

        norm = SymLogNorm(vmin=vmin, vmax=vmax, base=2, linthresh=2e-30)
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # Create main heatmap in appropriate grid position
    ax_main = plt.subplot(gs[-1, 0])  # Use -1 to always get last row
    main_hm = sns.heatmap(data, annot=False, cmap=cmap, fmt=fmt, ax=ax_main,
                         norm=norm, cbar=False)
    # Add node separator lines
    num_gpus = len(data.columns)
    
    # Add colorbar with metric units and title
    cax = plt.subplot(gs[:, 1])
    cbar = plt.colorbar(main_hm.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=6, pad=3)
        
    # Set colorbar title and label based on metric type
    metric_info = {
        'Temperature': {'unit': '°C', 'title': 'Temperature (°C)'},
        'Power': {'unit': 'W', 'title': 'Power (W)'},
        'Utilization': {'unit': '%', 'title': 'Utilization (%)'},
        'Clock': {'unit': 'GHz', 'title': 'Frequency (GHz)'},
        'Throughput': {'unit': 'GB/s', 'title': 'Throughput (GB/s)'},
        'Total': {'unit': 'GB' if 'GB' in title else 'MB', 'title': 'Rx Total (GB)' if 'GB' in title else 'Rx Total (MB)'},
    }
    
    for key, info in metric_info.items():
        if key in title:
            # Remove title setting since we'll add it on the right
            cbar.set_label(info['unit'], fontsize=9, labelpad=5)
            
            # Add rotated title on the right side
            if key == 'Total':
                title_text = f"{info['title']}"
            else:
                title_text = f"{'Peak' if peak else 'Avg'} {info['title']}"
            title_text = ' '
            cax.text(4.0, 0.5, title_text, 
                    rotation=90, 
                    fontsize=14,
                    transform=cax.transAxes,
                    va='center')
            break
        

    # Add borders around the colorbar
    cax.axhline(y=0, color='black', linewidth=1)
    cax.axhline(y=1, color='black', linewidth=1)
    cax.axvline(x=0, color='black', linewidth=1)
    cax.axvline(x=1, color='black', linewidth=1)

        
    # Create column averages heatmap only if needed
    if show_avg:
        ax_col = plt.subplot(gs[0, 0])
        sns.heatmap(pd.DataFrame(col_means).T, annot=False, cmap=cmap, fmt=fmt,
                    cbar=False, xticklabels=False, ax=ax_col, norm=norm)
        
        # Add borders around the average heatmap
        ax_col.axhline(y=0, color='black', linewidth=1)
        ax_col.axhline(y=1, color='black', linewidth=1)
        ax_col.axvline(x=0, color='black', linewidth=1)
        ax_col.axvline(x=len(data.columns), color='black', linewidth=1)
        
        # Set average row label
        ax_col.set_yticks([0.5])
        ax_col.set_yticklabels(['Per-GPU Avg.'], fontsize=11, rotation=0)
        
        # Add node separators to average row
        for i in range(8, num_gpus, 8):
            ax_col.axvline(x=i, color='black', linewidth=1)
    
    # Add border around the main heatmap
    ax_main.axhline(y=0, color='black', linewidth=1)
    ax_main.axhline(y=len(data), color='black', linewidth=1)
    ax_main.axvline(x=0, color='black', linewidth=1)
    ax_main.axvline(x=len(data.columns), color='black', linewidth=1)
    
    for i in range(8, num_gpus, 8):
        ax_main.axvline(x=i, color='black', linewidth=1)
    
        
    # Make y-tick labels horizontal
    for label in ax_main.get_yticklabels():
        label.set_rotation(0)
        # Blue color
        # label.set_color('blue')

    
    # Add model separator lines
    prev_model = None
    for idx, config_name in enumerate(data.index):
        current_model = config_name.split('_')[0]
        if prev_model is not None and current_model != prev_model:
            ax_main.axhline(y=idx, color='black', linewidth=1)
        prev_model = current_model
    
    # Customize axes
    # ax_main.set_xlabel('GPU Index', fontsize=12)
    ax_main.set_ylabel('    ', fontsize=19)
    
    # Set x-ticks for every 4 GPUs
    ax_main.set_xticks(np.arange(0, num_gpus, 4))
    # ax_main.set_xticklabels([f'{i}' for i in range(0, num_gpus, 4)], fontsize=12, rotation=0)
    # Remove xticklabels
    ax_main.set_xticklabels([])
    
    
    # Add node numbers on top if not showing average row
    if not show_avg:
        ax2 = ax_main.twiny()
    else:
        ax2 = ax_col.twiny()
        
    num_nodes = (num_gpus + 7) // 8
    node_positions = np.arange(4, num_gpus, 8)
    node_labels = [f'Node {i+1}' for i in range(num_nodes)]
    ax2.set_xlim(ax_main.get_xlim())
    ax2.set_xticks(node_positions)
    ax2.set_xticklabels(node_labels, fontsize=12)
    
    # Add colorbar
    cax = plt.subplot(gs[:, 1])
    cbar = plt.colorbar(main_hm.collections[0], cax=cax)
    cbar.ax.tick_params(labelsize=12)

    # Adjust layout with extra space for colorbar
    plt.tight_layout(rect=[0, 0, 0.95, 1])  # Added right margin for colorbar
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def main():
    parser = argparse.ArgumentParser(description='Generate heatmaps for GPU metrics')
    parser.add_argument('--models', nargs='+', 
                       default=['gpt3-175b', 'llama3-70b', 'mixtral-8x7b', 'mixtral-8x22b'],
                       help='List of models to include')
    args = parser.parse_args()


    # Define metrics dictionary
    metrics = {
        'thermal': {'gpu': 'GPU Temperature (°C); Lower is better', 'mem': 'Memory Temperature (°C); Lower is better'},
        'nvlink_bytes': {'rx': 'NVLink Receive Total (GB); Lower is better', 'tx': 'NVLink Transmit Total (GB); Lower is better'},
        'pcie_bytes': {'rx': 'PCIe Receive Total (GB); Lower is better', 'tx': 'PCIe Transmit Total (GB); Lower is better'}
    }


    # Define base paths
    base_dir = "${CHARLLM_ROOT}/CharLLM-PPT/results/optimization-sweep-h200-fsdp"
    figure_base_dir = "${CHARLLM_ROOT}/CharLLM-PPT/figures/heatmaps-fsdp"
    os.makedirs(figure_base_dir, exist_ok=True)

    # Process each metric type
    for metric_type, subtypes in metrics.items():
        for subtype, label in subtypes.items():
            try:
                # Collect data first
                combined_data = []
                combined_names = []
                combined_exec_times = []

                # Process each model
                for model in args.models:
                    model_dir = os.path.join(base_dir, model)
                    pattern = re.compile(f"^{model}_.*_00000$")
                    experiment_dirs = sorted([d for d in glob.glob(os.path.join(model_dir, "*")) 
                                        if os.path.isdir(d) and pattern.match(os.path.basename(d))])

                    if not experiment_dirs:
                        print(f"No experiments found for model {model}")
                        continue
                    
                    # Process each experiment for this model
                    for exp_dir in experiment_dirs:
                        row_avg = []
                        row_peak = []
                        has_missing_data = False
                        
                        # Get parallel config info
                        dir_name = os.path.basename(exp_dir)
                        parts = dir_name.split('_')
                        parallel_idx = parts[1]
                        
                        # Get strategy and format name
                        strategy = get_parallel_strategy(model, parallel_idx)
                        
                        config_name = f"{format_model_name(model)}_{format_strategy_name(strategy)}"

                        # Collect GPU metrics for both avg and peak
                        for gpu_idx in range(32):
                            avg_values = calculate_average_metric(exp_dir, metric_type, gpu_idx, peak=False)
                            peak_values = calculate_average_metric(exp_dir, metric_type, gpu_idx, peak=True)
                            
                            if avg_values is None or peak_values is None:
                                has_missing_data = True
                                break
                                
                            row_avg.append(avg_values[subtype])
                            row_peak.append(peak_values[subtype])

                        if not has_missing_data:
                            combined_data.append((row_avg, row_peak))
                            combined_names.append(config_name)
                            
                            # Get execution time
                            time = calculate_execution_time_from_profile(exp_dir)
                            combined_exec_times.append(time if time is not None else np.nan)

                if not combined_data:
                    print(f"No valid data found for {metric_type}_{subtype}")
                    continue

                # Create separate DataFrames for avg and peak
                df_avg = pd.DataFrame([d[0] for d in combined_data],
                                    index=combined_names,
                                    columns=[f'{i}' for i in range(32)])
                df_peak = pd.DataFrame([d[1] for d in combined_data],
                                     index=combined_names,
                                     columns=[f'{i}' for i in range(32)])
                                
                # Build set of valid config names
                valid_config_names = set()
                for model in args.models:
                    for parallel_idx in range(1, 10):  # adjust range as needed for your indices
                        strategy = get_parallel_strategy(model, parallel_idx)
                        if not strategy.startswith('unknown'):
                            config_name = f"{format_model_name(model)}_{format_strategy_name(strategy)}"
                            valid_config_names.add(config_name)

                # Filter DataFrames to only include valid strategies
                df_avg = df_avg[df_avg.index.isin(valid_config_names)]
                df_peak = df_peak[df_peak.index.isin(valid_config_names)]
                
                # Generate both average and peak heatmaps
                for is_peak, df in [(False, df_avg), (True, df_peak)]:
                    peak_suffix = '_peak' if is_peak else ''
                    output_path = os.path.join(figure_base_dir, 
                                             f"combined_{metric_type}_{subtype}{peak_suffix}_heatmap.pdf")
                    
                    title = f'Combined Models - {label}'
                    if metric_type == 'pcie_throughput':
                        create_heatmap(df, title, output_path, cmap='Blues', fmt='.2f',
                                     peak=is_peak)
                        
                    elif metric_type == 'clock':
                        create_heatmap(df, title, output_path, cmap='Blues', fmt='.2f',
                                     peak=is_peak)
                    elif metric_type == 'util' or metric_type == 'ib_bytes' or metric_type == 'pcie_bytes':
                        create_heatmap(df, title, output_path, cmap='Blues',
                                     peak=is_peak)
                        
                    elif metric_type == 'nvlink_bytes':
                        create_heatmap(df, title, output_path, cmap='Greens',
                                     peak=is_peak)
                        
                    elif metric_type == 'power':
                        create_heatmap(df, title, output_path, fmt='.0f',
                                     peak=is_peak)
                    else:
                        create_heatmap(df, title, output_path, peak=is_peak)
                    
                    print(f"Generated {'peak' if is_peak else 'average'} heatmap for {metric_type}_{subtype}")

            except Exception as e:
                print(f"Error generating heatmap for {metric_type}_{subtype}: {str(e)}")
                continue

    print(f"Processing complete! Combined heatmaps saved in {figure_base_dir}")
    
if __name__ == "__main__":
    main()