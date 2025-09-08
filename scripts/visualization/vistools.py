import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define metrics to plot with their properties
metrics = {
    'gpu_power': {
        'metric_type': 'power',
        'subtype': 'gpu',
        'title': 'GPU Power',
        'y1_label': 'Power\n(W)',
        'y2_label': 'Std. Dev.',
        'peak': True
    },
    'mem_power': {
        'metric_type': 'power',
        'subtype': 'mem',
        'title': 'GPU Memory Power',
        'y1_label': 'Power (W)',
        'y2_label': 'Std. Dev.',
        'peak': True
    },
    'gpu_util': {
        'metric_type': 'util',
        'subtype': 'gpu',
        'title': 'GPU Utilization',
        'y1_label': 'Utilization (%)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'mem_util': {
        'metric_type': 'util',
        'subtype': 'mem',
        'title': 'GPU Memory Utilization',
        'y1_label': 'Utilization (%)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'gpu_thermal': {
        'metric_type': 'thermal',
        'subtype': 'gpu',
        'title': 'GPU Temperature',
        'y1_label': 'Temperature\n(°C)',
        'y2_label': 'Std. Dev.',
        'peak': True
    },
    'mem_thermal': {
        'metric_type': 'thermal',
        'subtype': 'mem',
        'title': 'GPU Memory Temperature',
        'y1_label': 'Temperature (°C)',
        'y2_label': 'Std. Dev.',
        'peak': True
    },
    'gpu_clock': {
        'metric_type': 'clock',
        'subtype': 'gpu',
        'title': 'SM Clock',
        'y1_label': 'SM Freq.\n(GHz)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'ib_bytes': {
        'metric_type': 'ib_bytes',
        'subtype': 'rx',
        'title': 'IB Receive',
        'y1_label': 'Total (MB)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'pcie_throughput': {
        'metric_type': 'pcie_throughput',
        'subtype': 'rx',
        'title': 'PCIe Receive Throughput',
        'y1_label': 'Throughput (GB/s)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'pcie_bytes': {
        'metric_type': 'pcie_bytes',
        'subtype': 'rx',
        'title': 'PCIe Receive',
        'y1_label': 'PCIe Rx\n(GB)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
    'nvlink_bytes': {
        'metric_type': 'nvlink_bytes',
        'subtype': 'rx',
        'title': 'NVLink Receive',
        'y1_label': 'NVLink Rx\n(GB)',
        'y2_label': 'Std. Dev.',
        'peak': False
    },
}

def get_parallel_strategy(model_name, parallel_idx, gpu_type='h100'):
    """Map parallel index to actual parallelization strategy"""
    strategies = {
        'gpt3-175b': {
            '1': 'TP2-PP16',
            '2': 'TP4-PP8',
            '3': 'TP8-PP4',
            '4': 'TP1-PP32'
        },
        'llama3-70b': {
            '1': 'TP8-PP2',
            '2': 'TP4-PP4',
            '3': 'TP2-PP8',
            '4': 'TP1-PP16'
        },
        'mixtral-8x7b': {
            '1': 'EP4-TP2-PP1',
            '2': 'EP8-TP1-PP1'
        },
        'mixtral-8x22b': {
            '1': 'EP8-TP1-PP4',
            '2': 'EP8-TP2-PP2',
            '3': 'EP8-TP4-PP1'
        }
    }
    
    return strategies.get(model_name, {}).get(str(parallel_idx), f'unknown_{parallel_idx}')

def calculate_average_metric(exp_dir, metric_type, gpu_idx, peak=False):
    """Calculate average value for a given metric and GPU with proper unit conversions"""
    prefixes = {
        'power': ('gpu_power', 'mem_power'),
        'util': ('gpu_util', 'mem_util'),
        'thermal': ('gpu_thermal', 'mem_thermal'),
        'clock': ('sm_clock', 'mem_clock'),
        'ib_bytes': ('system_ib_rx_bytes_rank', 'system_ib_tx_bytes_rank'),
        'nvlink_bytes': ('nvlink_rx_throughput', 'nvlink_tx_throughput'),
        'pcie_throughput': ('pcie_rx_throughput', 'pcie_tx_throughput'),
        'pcie_bytes': ('pcie_rx_bytes', 'pcie_tx_bytes')
    }
    
    def calculate_weighted_average(df):
        """Helper function to calculate time-weighted average"""
        timestamps = df.iloc[:, 0].values
        values = df.iloc[:, 1].values
        time_intervals = np.diff(timestamps)
        
        # Use values[:-1] since we have one fewer interval than points
        return np.sum(values[:-1] * time_intervals) / np.sum(time_intervals)

    try:
        # Add helper function to safely read CSV
        def safe_read_csv(filepath):
            try:
                # First check if file exists and is not empty
                if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                    return None
                    
                df = pd.read_csv(filepath, header=None)
                
                # Check if DataFrame has the expected structure
                if df.empty or df.shape[1] < 2:  # We expect at least 2 columns
                    return None
                    
                # Check if second column has any non-null values
                if df.iloc[:, 1].isna().all():
                    return None
                    
                return df
                
            except (pd.errors.EmptyDataError, pd.errors.ParserError, IndexError):
                return None
            
        if metric_type == 'ib_bytes':
            # For IB metrics, we only have one value per node (every 8 GPUs)
            # Use the first GPU in each node as the representative
            node_representative_gpu = (gpu_idx // 8) * 8
            rx_prefix, tx_prefix = prefixes[metric_type]
            
            # Find all IB interfaces for this node
            rx_files = sorted(glob.glob(f'{exp_dir}/{rx_prefix}_{node_representative_gpu}_id_ib*.csv'))
            tx_files = sorted(glob.glob(f'{exp_dir}/{tx_prefix}_{node_representative_gpu}_id_ib*.csv'))
            
            if not rx_files or not tx_files:
                return None
                
            # Sum up data from all IB interfaces for this node
            rx_total = 0
            tx_total = 0
            
            for rx_file, tx_file in zip(rx_files, tx_files):
                rx_data = safe_read_csv(rx_file)
                tx_data = safe_read_csv(tx_file)
                
                # Check if either file read failed
                if rx_data is None or tx_data is None:
                    return None
                
                # Check if files are empty
                if rx_data.empty or tx_data.empty:
                    return None
                
                # Calculate deltas considering uint32 overflow
                def calculate_delta_with_overflow(values, max_uint32=4294967295):
                    deltas = []
                    for i in range(1, len(values)):
                        if values[i] < values[i-1]:
                            delta = (max_uint32 - values[i-1]) + values[i]
                        else:
                            delta = values[i] - values[i-1]
                        deltas.append(delta)
                    return deltas
                
                # Calculate total bytes transferred
                rx_bytes = rx_data.iloc[:, 1].values
                tx_bytes = tx_data.iloc[:, 1].values
                
                rx_deltas = calculate_delta_with_overflow(rx_bytes)
                tx_deltas = calculate_delta_with_overflow(tx_bytes)
                
                rx_total += sum(rx_deltas) / (1024 * 1024)  # Convert to MB
                tx_total += sum(tx_deltas) / (1024 * 1024)  # Convert to MB
            
            # Return same value for all GPUs in the same node
            return {
                'rx': rx_total,
                'tx': tx_total
            }
        elif metric_type == 'nvlink_bytes' or metric_type == 'pcie_bytes':
            rx_prefix, tx_prefix = prefixes[metric_type]
            rx_data = safe_read_csv(f'{exp_dir}/{rx_prefix}_{gpu_idx}.csv')
            tx_data = safe_read_csv(f'{exp_dir}/{tx_prefix}_{gpu_idx}.csv')
            
            # Check if either file read failed
            if rx_data is None or tx_data is None:
                return None
            
            # Check if files are empty
            if rx_data.empty or tx_data.empty:
                return None
            
            # Calculate deltas considering uint32 overflow
            rx_bytes = rx_data.iloc[:, 1].values
            tx_bytes = tx_data.iloc[:, 1].values
            
                                
            # Calculate deltas considering uint32 overflow
            def calculate_delta_with_overflow(values, max_uint32=4294967295):
                deltas = []
                for i in range(1, len(values)):
                    if values[i] < values[i-1]:
                        delta = (max_uint32 - values[i-1]) + values[i]
                    else:
                        delta = values[i] - values[i-1]
                    deltas.append(delta)
                return deltas
            
            rx_deltas = calculate_delta_with_overflow(rx_bytes)
            tx_deltas = calculate_delta_with_overflow(tx_bytes)
            
            # Sum up total bytes transferred
            rx_total = sum(rx_deltas)
            tx_total = sum(tx_deltas)
            
            # Convert to GB
            if metric_type == 'nvlink_bytes':
                conversion = 1024 * 1024
            else:
                conversion = 1024 * 1024 * 1024  # Bytes to GB
            return {
                'rx': rx_total / conversion,
                'tx': tx_total / conversion
            }
        elif metric_type in ['power', 'util', 'thermal']:
            gpu_prefix, mem_prefix = prefixes[metric_type]
            # Try both gpu_power_X.csv and power_X.csv for GPU
            gpu_data = None
            for prefix in [gpu_prefix, 'power']:
                gpu_data = safe_read_csv(f'{exp_dir}/{prefix}_{gpu_idx}.csv')
                if gpu_data is not None:
                    break
            # Try both mem_power_X.csv and power_X.csv for MEM
            mem_data = None
            for prefix in [mem_prefix, 'power']:
                mem_data = safe_read_csv(f'{exp_dir}/{prefix}_{gpu_idx}.csv')
                if mem_data is not None:
                    break

            # Check if either file read failed
            if gpu_data is None or mem_data is None:
                return None

            # Check if files are empty
            if gpu_data.empty or mem_data.empty:
                return None

            if peak:
                return {
                    'gpu': gpu_data.iloc[:, 1].max(),  # Peak value
                    'mem': mem_data.iloc[:, 1].max()   # Peak value
                }
            else:
                return {
                    'gpu': calculate_weighted_average(gpu_data),
                    'mem': calculate_weighted_average(mem_data)
                }
        elif metric_type == 'clock':
            gpu_prefix, mem_prefix = prefixes[metric_type]
            gpu_data = safe_read_csv(f'{exp_dir}/{gpu_prefix}_{gpu_idx}.csv')
            mem_data = safe_read_csv(f'{exp_dir}/{mem_prefix}_{gpu_idx}.csv')
            
            # Check if either file read failed
            if gpu_data is None or mem_data is None:
                return None
            
            # Check if files are empty
            if gpu_data.empty or mem_data.empty:
                return None
            
            return {
                    'gpu': calculate_weighted_average(gpu_data) / 1000,  # Convert to GHz
                    'mem': calculate_weighted_average(mem_data) / 1000
            }
        elif metric_type == 'pcie_throughput':
            rx_prefix, tx_prefix = prefixes[metric_type]
            rx_data = safe_read_csv(f'{exp_dir}/{rx_prefix}_{gpu_idx}.csv')
            tx_data = safe_read_csv(f'{exp_dir}/{tx_prefix}_{gpu_idx}.csv')
            
            # Check if either file read failed
            if rx_data is None or tx_data is None:
                return None
            
            # Check if files are empty
            if rx_data.empty or tx_data.empty:
                return None
            
            return {
                'rx': calculate_weighted_average(rx_data) / (1024 * 1024),  # KB/s to GB/s
                'tx': calculate_weighted_average(tx_data) / (1024 * 1024)   # KB/s to GB/s
            }
        else:
            rx_prefix, tx_prefix = prefixes[metric_type]
            rx_data = safe_read_csv(f'{exp_dir}/{rx_prefix}_{gpu_idx}.csv')
            tx_data = safe_read_csv(f'{exp_dir}/{tx_prefix}_{gpu_idx}.csv')
            
            # Check if either file read failed
            if rx_data is None or tx_data is None:
                return None
            
            # Check if files are empty
            if rx_data.empty or tx_data.empty:
                return None
            
            return {
                'rx': calculate_weighted_average(rx_data),
                'tx': calculate_weighted_average(tx_data)
            }
    except FileNotFoundError:
        return None

def calculate_total_energy(exp_dir):
    """Calculate total energy consumption from GPU and memory power data"""
    total_energy = 0
    
    if 'h100' in exp_dir:
        num_gpus=64
    else:
        num_gpus=32
    
    # Sum energy across all GPUs
    for gpu_idx in range(num_gpus):
        # Try both possible file names for GPU power
        gpu_file = None
        mem_file = None
        for prefix in ['gpu_power', 'power']:
            candidate = os.path.join(exp_dir, f'{prefix}_{gpu_idx}.csv')
            if os.path.exists(candidate):
                gpu_file = candidate
                break
        for prefix in ['mem_power', 'power']:
            candidate = os.path.join(exp_dir, f'{prefix}_{gpu_idx}.csv')
            if os.path.exists(candidate):
                mem_file = candidate
                break

        if not gpu_file or not mem_file:
            continue

        try:
            gpu_data = pd.read_csv(gpu_file, header=None)
            gpu_timestamps = gpu_data.iloc[:, 0].to_numpy()
            gpu_power = gpu_data.iloc[:, 1].to_numpy()
            gpu_energy = np.trapz(gpu_power, gpu_timestamps)
            total_energy += gpu_energy
        except:
            continue

    return total_energy

def calculate_execution_time_from_profile(exp_dir):
    """Calculate execution time from profiling data"""
    # Get power data from the main rank, and infer the execution time
    file_path = os.path.join(exp_dir, f'gpu_power_0.csv')
    # If file does not exist, try power_0.csv
    if not os.path.exists(file_path):
        file_path = os.path.join(exp_dir, f'power_0.csv')
    
    if not os.path.exists(file_path):
        return None
        
    try:
        df = pd.read_csv(file_path, header=None)
        if len(df) < 2:
            return None
            
        # Use entire time range without skipping
        start_time = df.iloc[0, 0]
        end_time = df.iloc[-1, 0]
        exec_time = (end_time - start_time)
        
        # Sanity check - execution time should be positive
        if exec_time > 0:
            return exec_time
    except:
        return None
    
    return None