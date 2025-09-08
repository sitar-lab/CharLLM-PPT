"""Monitor the power usage of GPUs."""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import tempfile
import typing

# CharLLM-PPT: Add imports for polling
from multiprocessing import Process, Queue, Value
from queue import Empty
from time import sleep, time

import pandas as pd

# CharLLM-PPT: Add imports for system metrics
import requests, traceback
from sklearn.metrics import auc

from zeus.device import get_gpus
from zeus.device.gpu.amd import amdsmi_is_available
from zeus.device.gpu.nvidia import nvml_is_available

from zeus.utils.logging import get_logger

# Only import monitoring dependencies if enabled
if nvml_is_available():
    import pynvml

    is_nvidia = True
    is_amd = False
elif amdsmi_is_available():
    is_amd = True
    is_nvidia = False
if os.environ.get("ZEUS_MONITOR_SYS", "").lower() in ("1", "true", "yes"):
    from prometheus_client.parser import text_string_to_metric_families


# CharLLM-PPT: Add SystemMetrics class to collect system-level metrics
class SystemMetrics:
    """Container for system-level metrics from node_exporter."""

    # {metric: (metric_name, label_patterns)}
    METRICS = {
        # CPU Metrics (H100/H200 HGX nodes has 2 sockets with 32 cores each)
        "cpu_usage": (
            "node_cpu_seconds",
            {
                "cpu": "^([0-9]|[12][0-9]|3[0-9]|[45][0-9]|6[0-3])$",
                "mode": "user|system|irq|softirq|nice",
            },
        ),
        "cpu_idle": (
            "node_cpu_seconds",
            {"cpu": "^([0-9]|[12][0-9]|3[0-9]|[45][0-9]|6[0-3])$", "mode": "idle"},
        ),
        "cpu_iowait": (
            "node_cpu_seconds",
            {"cpu": "^([0-9]|[12][0-9]|3[0-9]|[45][0-9]|6[0-3])$", "mode": "iowait"},
        ),
        "cpu_temp": (
            "node_thermal_zone_temp",
            {"zone": "^([0-9])$", "type": "x86_pkg_temp"},
        ),
        # Memory Metrics
        "memory_used": "node_memory_MemTotal_bytes",
        "memory_free": "node_memory_MemFree_bytes",
        # InfiniBand Metrics
        "ib_rx_bytes": ("node_network_receive_bytes", {"device": "^ib[0-9]+$"}),
        "ib_tx_bytes": ("node_network_transmit_bytes", {"device": "^ib[0-9]+$"}),
    }

    def __init__(self) -> None:
        """Initialize system metrics storage."""
        self.data = {metric: [] for metric in self.METRICS}
        self.node_exporter_url = "http://localhost:9100/metrics"

    def _match_labels(self, sample_labels: dict, required_labels: dict) -> bool:
        """Check if sample labels match the required pattern."""
        import re

        for key, pattern in required_labels.items():
            if key not in sample_labels:
                return False
            if not re.match(pattern, sample_labels[key]):
                return False
        return True

    def collect_metrics(self, current_time: float) -> None:
        """Collect all system metrics at current timestamp."""
        try:
            response = requests.get(self.node_exporter_url)
            if response.status_code != 200:
                print(f"Error fetching metrics: {response.status_code}")
                return

            metrics = text_string_to_metric_families(response.text)

            for metric in metrics:
                for key, spec in self.METRICS.items():
                    if isinstance(spec, tuple):
                        metric_name, label_patterns = spec
                        if metric.name == metric_name:
                            for sample in metric.samples:
                                # print(f"Sample: {sample}")
                                if self._match_labels(sample.labels, label_patterns):
                                    device = sample.labels.get("device", "")
                                    cpu = sample.labels.get("cpu", "")
                                    zone = sample.labels.get("zone", "")

                                    # Store the label information along with the data
                                    self.data[key].append(
                                        [
                                            current_time,
                                            sample.value,
                                            device
                                            or cpu
                                            or zone,  # Store identifying label
                                        ]
                                    )
                    elif metric.name == spec:  # Simple metrics without label matching
                        for sample in metric.samples:
                            self.data[key].append([current_time, sample.value])

        except Exception as e:
            print(f"Error collecting system metrics: {e}")
            import traceback

            traceback.print_exc()

    def save_to_csv(self, base_path: str, global_rank: str) -> None:
        """Save metrics to CSV files."""
        for metric in self.METRICS:
            if self.data[metric]:  # Only save if data exists
                if isinstance(self.METRICS[metric], tuple):  # Metrics with labels
                    # Group by device/socket/zone
                    groups = {}
                    for row in self.data[metric]:
                        identifier = row[2]  # device/cpu/zone identifier
                        if identifier not in groups:
                            groups[identifier] = []
                        groups[identifier].append([row[0], row[1]])  # time and value

                    # Save separate file for each group
                    for identifier, values in groups.items():
                        filepath = os.path.join(
                            base_path,
                            f"system_{metric}_rank_{global_rank}_id_{identifier}.csv",
                        )
                        with open(filepath, "w") as f:
                            for row in values:
                                f.write(",".join(map(str, row)) + "\n")
                else:  # Simple metrics
                    filepath = os.path.join(
                        base_path, f"system_{metric}_{global_rank}.csv"
                    )
                    with open(filepath, "w") as f:
                        for row in self.data[metric]:
                            f.write(",".join(map(str, row)) + "\n")


# CharLLM-PPT: Add GPUMetrics class to collect GPU-related metrics
class GPUMetrics:
    """Container for GPU metrics with unified data management."""

    if is_nvidia:

        # Map metrics to their NVML field IDs and divisors
        NVML_FIELD_METRICS = {
            "gpu_power": (pynvml.NVML_FI_DEV_POWER_INSTANT, 1000),
            "mem_power": [
                (pynvml.NVML_FI_DEV_POWER_AVERAGE, pynvml.NVML_POWER_SCOPE_MEMORY),
                1000,
            ],
            "mem_thermal": (pynvml.NVML_FI_DEV_MEMORY_TEMP, 1),
            "pcie_tx_bytes": (pynvml.NVML_FI_DEV_PCIE_COUNT_TX_BYTES, 1),
            "pcie_rx_bytes": (pynvml.NVML_FI_DEV_PCIE_COUNT_RX_BYTES, 1),
            "nvlink_tx_throughput": (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_TX, 1),
            "nvlink_rx_throughput": (pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_RAW_RX, 1),
        }

        # Metrics that need special handling through other NVML calls
        SPECIAL_METRICS = {
            "gpu_util": "getGPUUtilization",
            "mem_util": "getMemoryUtilization",
            "gpu_thermal": "getTemperature",
            "sm_clock": "getSMClock",
            "mem_clock": "getMemoryClock",
            "pcie_tx_throughput": "getPCIeTxThroughput",
            "pcie_rx_throughput": "getPCIeRxThroughput",
        }

        ALL_METRICS = list(NVML_FIELD_METRICS.keys()) + list(SPECIAL_METRICS.keys())

    if is_amd:

        AMD_METRICS = {
            "gpu_util": "getGPUUtilization",
            "mem_util": "getMemoryUtilization",
            "gpu_power": "getInstantPowerUsage",
            "gpu_thermal": "getTemperature",
            "mem_thermal": "getMemTemperature",
            "sm_clock": "getSMClock",
            "mem_clock": "getMemoryClock",
        }

        ALL_METRICS = AMD_METRICS
        SPECIAL_METRICS = AMD_METRICS

    def __init__(self, gpu_indices: list[int]) -> None:
        """Initialize metrics storage."""
        self.gpu_indices = gpu_indices
        self.raw_data = {metric: [] for metric in self.ALL_METRICS}

        # Pre-prepare field values array
        if is_nvidia:
            self.field_values = []
            for metric, spec in self.NVML_FIELD_METRICS.items():
                if isinstance(spec[0], list):
                    self.field_values.append((spec[0][0], spec[0][1]))
                else:
                    self.field_values.append(spec[0])

    def collect_metrics(self, gpus, current_time: float) -> None:
        """Collect all metrics at current timestamp using batched calls."""
        for idx in self.gpu_indices:
            gpu = gpus.gpus[idx]

            # Store raw results without processing
            if is_nvidia:
                results = pynvml.nvmlDeviceGetFieldValues(gpu.handle, self.field_values)
                for metric, result in zip(self.NVML_FIELD_METRICS.keys(), results):
                    self.raw_data[metric].append((current_time, result))

            # Handle special metrics - store raw values
            for metric, getter in self.SPECIAL_METRICS.items():
                try:
                    method = getattr(gpu, getter)
                    value = method()
                    self.raw_data[metric].append((current_time, value))
                except Exception as e:
                    print(f"Error collecting {metric} for GPU {idx}: {e}")
                    self.raw_data[metric].append((current_time, -1))

    def save_to_csv(self, base_path: str, global_rank: str) -> None:
        """Process raw data and save metrics to CSV files."""
        processed_data = {}

        # Process NVML field metrics
        if is_nvidia:
            for metric, spec in self.NVML_FIELD_METRICS.items():
                processed_data[metric] = []
                for time, result in self.raw_data[metric]:
                    if result.nvmlReturn == pynvml.NVML_SUCCESS:
                        value = result.value.uiVal / spec[1]
                    else:
                        value = -1
                    processed_data[metric].append([time, value])

        # Process special metrics
        for metric in self.SPECIAL_METRICS:
            processed_data[metric] = [
                [time, value] for time, value in self.raw_data[metric]
            ]

        # Save to files
        for metric in self.ALL_METRICS:
            if not base_path.endswith(".csv"):
                filepath = os.path.join(base_path, f"{metric}_{global_rank}.csv")
            else:
                filepath = base_path.replace(".csv", f"_{metric}_{global_rank}.csv")

            with open(filepath, "w") as f:
                for row in processed_data[metric]:
                    f.write(",".join(map(str, row)) + "\n")


class PowerMonitor:
    """Monitor power usage from GPUs.

    This class acts as a lower level wrapper around a Python process that polls
    the power consumption of GPUs. This is primarily used by
    [`ZeusMonitor`][zeus.monitor.ZeusMonitor] for older architecture GPUs that
    do not support the nvmlDeviceGetTotalEnergyConsumption API.

    !!! Warning
        Since the monitor spawns a child process, **it should not be instantiated as a global variable**.
        Python puts a protection to prevent creating a process in global scope.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    Attributes:
        gpu_indices (list[int]): Indices of the GPUs to monitor.
        update_period (int): Update period of the power monitor in seconds.
            Holds inferred update period if `update_period` was given as `None`.
    """

    def __init__(
        self,
        global_rank: int,
        gpu_indices: list[int] | None = None,
        update_period: float | None = None,
        csv_path: str | None = None,
    ) -> None:
        """Initialize the power monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the power monitor in seconds. If None,
                infer the update period by max speed polling the power counter for
                each GPU model.
            csv_path: If given, the power polling process will write measurements
                to this path. Otherwise, a temporary file will be used.
        """
        if gpu_indices is not None and not gpu_indices:
            raise ValueError("`gpu_indices` must be either `None` or non-empty")

        # Get GPUs
        gpus = get_gpus()

        # Set up logging.
        self.logger = get_logger(type(self).__name__)

        # Set global rank
        self.global_rank = global_rank

        # Set output directory for CSV files
        self.csv_path = csv_path

        # Get GPUs
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(len(gpus)))
        )
        self.logger.info("Monitoring power usage of GPUs %s", self.gpu_indices)

        # Infer the update period if necessary.
        if update_period is None:
            update_period = infer_counter_update_period(self.gpu_indices)
        self.update_period = update_period

        # Only collect system metrics if GPU 0 is monitored AND env var is set
        self.collect_sys_metrics = gpu_indices[0] == 0 and os.environ.get(
            "ZEUS_MONITOR_SYS", ""
        ).lower() in ("1", "true", "yes")
        # Replace all the individual CSV file handling with:
        self.metrics = GPUMetrics(self.gpu_indices)

        # Only the first rank within a node will collect system metrics
        self.system_metrics = SystemMetrics() if self.collect_sys_metrics else None

        # Create a shared event for monitoring window state
        self.monitoring_event = mp.Event()

        # Create separate queues for GPU and system metrics
        self.gpu_metrics_queue = Queue()
        self.system_metrics_queue = Queue() if self.collect_sys_metrics else None

        # Add initialization flags
        self.init_ready = Value("i", 0)  # Flag for process initialization

        # Create GPU metrics process
        self.gpu_process = Process(
            target=_gpu_polling_process,
            args=(
                self.gpu_indices,
                self.gpu_metrics_queue,
                self.update_period,
                self.monitoring_event,
                self.init_ready,
            ),
        )

        # Create system metrics process only for rank with GPU 0
        self.system_process = None
        if self.collect_sys_metrics:
            self.system_process = Process(
                target=_system_polling_process,
                args=(
                    self.system_metrics_queue,
                    self.update_period,
                    self.monitoring_event,
                ),
            )

        # Start processes
        self.gpu_process.start()

        # Wait for initialization
        timeout = 300  # 300 second timeout
        start_time = time()
        while time() - start_time < timeout:
            if self.init_ready.value:
                self.logger.info("GPU polling process initialized successfully")
                break
            sleep(0.1)
        else:
            raise TimeoutError("Timeout waiting for GPU polling process to initialize")

        if self.system_process:
            self.system_process.start()

        # Register _stop to be called at exit
        atexit.register(self._stop)

    def _cleanup_queue(self, queue) -> None:
        """Helper method to safely cleanup a queue."""
        if queue:
            try:
                while True:
                    queue.get_nowait()
            except Empty:
                pass
            queue.close()
            queue.join_thread()

    def _stop(self) -> None:
        """Stop all monitoring processes."""
        print("Stopping power monitor")

        # Stop GPU metrics process
        if self.gpu_process:
            print("\nSending termination signal to GPU process...")
            self.gpu_process.terminate()

            try:
                gpu_data = self.gpu_metrics_queue.get(timeout=300.0)
                if gpu_data:
                    self.metrics.raw_data = gpu_data
                    self.metrics.save_to_csv(self.csv_path, str(self.global_rank))
            except Empty:
                print("Timeout waiting for GPU data")

            self.gpu_process.join(timeout=30.0)
            if self.gpu_process.is_alive():
                self.gpu_process.kill()
            self.gpu_process = None

        # Stop system metrics process
        if self.system_process:
            print("\nSending termination signal to system metrics process...")
            self.system_process.terminate()

            try:
                system_data = self.system_metrics_queue.get(timeout=300.0)
                if system_data:
                    self.system_metrics.data = system_data
                    self.system_metrics.save_to_csv(
                        self.csv_path, str(self.global_rank)
                    )
            except Empty:
                print("Timeout waiting for system data")

            self.system_process.join(timeout=30.0)
            if self.system_process.is_alive():
                self.system_process.kill()
            self.system_process = None

        # Clean up queues
        self._cleanup_queue(self.gpu_metrics_queue)
        self._cleanup_queue(self.system_metrics_queue)


# CharLLM-PPT: polling processes for GPU metrics
def _gpu_polling_process(
    gpu_indices, metrics_queue, update_period, monitoring_event, init_ready
) -> None:
    """Run the GPU metrics polling process."""
    try:
        print(f"Starting GPU polling process for indices {gpu_indices}")
        gpus = get_gpus()
        gpu_metrics = GPUMetrics(gpu_indices)

        # Signal successful initialization
        init_ready.value = 1

        total_samples = 0

        # Set up signal handling
        import signal

        running = True

        def signal_handler(signum, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, signal_handler)

        while running:
            try:
                now = time()
                # Check environment variable
                if monitoring_event.is_set():
                    gpu_metrics.collect_metrics(gpus, now)
                    total_samples += 1

                    if total_samples % 100 == 0:
                        print(
                            f"Rank {gpu_indices} Collected {total_samples} GPU samples"
                        )

                if (sleep_time := update_period - (time() - now)) > 0:
                    sleep(sleep_time)
            except Exception as e:
                print(f"Error in GPU polling loop: {e}")
                traceback.print_exc()
                sleep(update_period)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received in GPU polling process")
    finally:
        print(f"\nGPU polling shutdown, samples collected: {total_samples}")
        try:
            metrics_queue.put(gpu_metrics.raw_data)
        except Exception as e:
            print(f"Error sending GPU data: {e}")
            traceback.print_exc()


# CharLLM-PPT: polling process for system metrics
def _system_polling_process(metrics_queue, update_period, monitoring_event) -> None:
    """Run the system metrics polling process."""
    try:
        system_metrics = SystemMetrics()
        total_samples = 0

        # Set up signal handling
        import signal

        running = True

        def signal_handler(signum, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGTERM, signal_handler)

        while running:
            try:
                now = time()
                if monitoring_event.is_set():
                    system_metrics.collect_metrics(now)
                    total_samples += 1

                    if total_samples % 100 == 0:
                        print(f"Collected {total_samples} System samples")

                if (sleep_time := update_period - (time() - now)) > 0:
                    sleep(sleep_time)
            except Exception as e:
                print(f"Error in system polling loop: {e}")
                traceback.print_exc()
                sleep(update_period)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt received in system polling process")
    finally:
        print(f"\nSystem polling shutdown, samples collected: {total_samples}")
        try:
            metrics_queue.put(system_metrics.data)
        except Exception as e:
            print(f"Error sending system data: {e}")
            traceback.print_exc()


# Legacy code below this point
def infer_counter_update_period(gpu_indicies: list[int]) -> float:
    """Infer the update period of the NVML power counter.

    NVML counters can update as slow as 10 Hz depending on the GPU model, so
    there's no need to poll them too faster than that. This function infers the
    update period for each unique GPU model and selects the fastest-updating
    period detected. Then, it returns half the period to ensure that the
    counter is polled at least twice per update period.
    """
    logger = get_logger(__name__)

    # get gpus
    gpus = get_gpus()

    # For each unique GPU model, infer the update period.
    update_period = 0.0
    gpu_models_covered = set()
    for index in gpu_indicies:
        if (model := gpus.getName(index)) not in gpu_models_covered:
            logger.info(
                "Detected %s, inferring NVML power counter update period.", model
            )
            gpu_models_covered.add(model)
            detected_period = _infer_counter_update_period_single(index)
            logger.info(
                "Counter update period for %s is %.2f s",
                model,
                detected_period,
            )
            update_period = min(update_period, detected_period)

    # Target half the update period to ensure that the counter is enough.
    update_period /= 2.0

    # Anything less than ten times a second is probably too slow.
    if update_period > 0.1:
        logger.warning(
            "Inferred update period (%.2f s) is too long. Using 0.1 s instead.",
            update_period,
        )
        update_period = 0.1
    return update_period


def _infer_counter_update_period_single(gpu_index: int) -> float:
    """Infer the update period of the NVML power counter for a single GPU."""
    # get gpus
    gpus = get_gpus()
    # Collect 1000 samples of the power counter with timestamps.
    time_power_samples: list[tuple[float, int]] = [(0.0, 0) for _ in range(1000)]
    for i in range(len(time_power_samples)):
        time_power_samples[i] = (
            time(),
            gpus.getInstantPowerUsage(gpu_index),
        )

    # Find the timestamps when the power readings changed.
    time_power_samples = time_power_samples[10:]
    changed_times = []
    prev_power = time_power_samples[0][1]
    for t, p in time_power_samples:
        if p != prev_power:
            changed_times.append(t)
            prev_power = p

    # Compute the minimum time difference between power change timestamps.
    intervals = [
        time2 - time1 for time1, time2 in zip(changed_times, changed_times[1:])
    ]
    if len(intervals) == 0:
        return 0.1
    return min(intervals)
