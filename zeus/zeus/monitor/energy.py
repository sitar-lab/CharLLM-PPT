"""Measure the GPU time and energy consumption of a block of code."""

from __future__ import annotations

import os
import warnings
from typing import Literal
from time import time, sleep
from pathlib import Path
from dataclasses import dataclass
from functools import cached_property

from zeus.monitor.power import PowerMonitor
from zeus.utils.logging import get_logger
from zeus.utils.framework import sync_execution as sync_execution_fn
from zeus.device import get_gpus, get_cpus
from zeus.device.gpu.common import ZeusGPUInitError, EmptyGPUs
from zeus.device.cpu.common import ZeusCPUInitError, ZeusCPUNoPermissionError, EmptyCPUs

logger = get_logger(__name__)


@dataclass
class Measurement:
    """Measurement result of one window.

    Attributes:
        time: Time elapsed (in seconds) during the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        # TODO: Update method to total_gpu_energy, which may cause breaking changes in the examples/
        return sum(self.gpu_energy.values())


@dataclass
class MeasurementState:
    """Measurement state to keep track of measurements in start_window.

    Used in ZeusMonitor to map string keys of measurements to this dataclass.

    Attributes:
        time: The beginning timestamp of the measurement window.
        gpu_energy: Maps GPU indices to the energy consumed (in Joules) during the
            measurement window. GPU indices are from the DL framework's perspective
            after applying `CUDA_VISIBLE_DEVICES`.
        cpu_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d). This can
            be 'None' if CPU measurement is not available.
        dram_energy: Maps CPU indices to the energy consumed (in Joules) during the measurement
            window. Each CPU index refers to one powerzone exposed by RAPL (intel-rapl:d)  and DRAM
            measurements are taken from sub-packages within each powerzone. This can be 'None' if
            CPU measurement is not available or DRAM measurement is not available.
    """

    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        return sum(self.gpu_energy.values())


class ZeusMonitor:
    """Measure the GPU energy and time consumption of a block of code.

    Works for multi-GPU and heterogeneous GPU types. Aware of `CUDA_VISIBLE_DEVICES`.
    For instance, if `CUDA_VISIBLE_DEVICES=2,3`, GPU index `1` passed into `gpu_indices`
    will be interpreted as CUDA device `3`.

    You can mark the beginning and end of a measurement window, during which the GPU
    energy and time consumed will be recorded. Multiple concurrent measurement windows
    are supported.

    For Volta or newer GPUs, energy consumption is measured very cheaply with the
    `nvmlDeviceGetTotalEnergyConsumption` API. On older architectures, this API is
    not supported, so a separate Python process is used to poll `nvmlDeviceGetPowerUsage`
    to get power samples over time, which are integrated to compute energy consumption.

    !!! Warning
        Since the monitor may spawn a process to poll the power API on GPUs older than
        Volta, **the monitor should not be instantiated as a global variable
        without guarding it with `if __name__ == "__main__"`**.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.

    ## Integration Example

    ```python
    import time
    from zeus.monitor import ZeusMonitor

    def training():
        # A dummy training function
        time.sleep(5)

    if __name__ == "__main__":
        # Time/Energy measurements for four GPUs will begin and end at the same time.
        gpu_indices = [0, 1, 2, 3]
        monitor = ZeusMonitor(gpu_indices)

        # Mark the beginning of a measurement window. You can use any string
        # as the window name, but make sure it's unique.
        monitor.begin_window("entire_training")

        # Actual work
        training()

        # Mark the end of a measurement window and retrieve the measurment result.
        result = monitor.end_window("entire_training")

        # Print the measurement result.
        print(f"Training consumed {result.total_energy} Joules.")
        for gpu_idx, gpu_energy in result.gpu_energy.items():
            print(f"GPU {gpu_idx} consumed {gpu_energy} Joules.")
    ```

    Attributes:
        gpu_indices (`list[int]`): Indices of all the CUDA devices to monitor, from the
            DL framework's perspective after applying `CUDA_VISIBLE_DEVICES`.
    """

    def __init__(
        self,
        global_rank: int,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        approx_instant_energy: bool = False,
        log_file: str | Path | None = None,
        sync_execution_with: Literal["torch", "jax"] = "torch",
    ) -> None:
        """Instantiate the monitor.

        Args:
            global_rank: The global rank of the process. This is used to determine the
                output file name for the log file.
            gpu_indices: Indices of all the CUDA devices to monitor. Time/Energy measurements
                will begin and end at the same time for these GPUs (i.e., synchronized).
                If None, all the GPUs available will be used. `CUDA_VISIBLE_DEVICES`
                is respected if set, e.g., GPU index `1` passed into `gpu_indices` when
                `CUDA_VISIBLE_DEVICES=2,3` will be interpreted as CUDA device `3`.
                `CUDA_VISIBLE_DEVICES`s formatted with comma-separated indices are supported.
            cpu_indices: Indices of the CPU packages to monitor. If None, all CPU packages will
                be used.
            approx_instant_energy: When the execution time of a measurement window is
                shorter than the NVML energy counter's update period, energy consumption may
                be observed as zero. In this case, if `approx_instant_energy` is True, the
                window's energy consumption will be approximated by multiplying the current
                instantaneous power consumption with the window's execution time. This should
                be a better estimate than zero, but it's still an approximation.
            log_file: Path to the log CSV file. If `None`, logging will be disabled.
            sync_execution_with: Deep learning framework to use to synchronize CPU/GPU computations.
                Defaults to `"torch"`, in which case `torch.cuda.synchronize` will be used.
                See [`sync_execution`][zeus.utils.framework.sync_execution] for more details.
        """
        # Save arguments.
        self.approx_instant_energy = approx_instant_energy
        self.sync_with: Literal["torch", "jax"] = sync_execution_with

        # Get GPU instances.
        try:
            self.gpus = get_gpus()
        except ZeusGPUInitError:
            self.gpus = EmptyGPUs()

        # Get CPU instance.
        try:
            self.cpus = get_cpus()
        except ZeusCPUInitError:
            self.cpus = EmptyCPUs()
        except ZeusCPUNoPermissionError as err:
            if cpu_indices:
                raise RuntimeError(
                    "Root privilege is required to read RAPL metrics. See "
                    "https://ml.energy/zeus/getting_started/#system-privileges "
                    "for more information or disable CPU measurement by passing cpu_indices=[] to "
                    "ZeusMonitor"
                ) from err
            self.cpus = EmptyCPUs()

        # Resolve GPU indices. If the user did not specify `gpu_indices`, use all available GPUs.
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(len(self.gpus)))
        )

        # Resolve CPU indices. If the user did not specify `cpu_indices`, use all available CPUs.
        self.cpu_indices = (
            cpu_indices if cpu_indices is not None else list(range(len(self.cpus)))
        )

        logger.info("Monitoring GPU indices %s for global rank %s.", self.gpu_indices, global_rank)
        logger.info("Monitoring CPU indices %s", self.cpu_indices)

        # Initialize loggers.
        if log_file is None:
            self.log_file = None
        else:
            if dir := os.path.dirname(log_file):
                os.makedirs(dir, exist_ok=True)
            self.log_file = open(log_file, "w")
            logger.info("Writing measurement logs to %s.", log_file)
            self.log_file.write(
                f"start_time,window_name,elapsed_time,{','.join(map(lambda i: f'gpu{i}_energy', self.gpu_indices))}\n",
            )
            self.log_file.flush()

        # Initialize power monitors for older architecture GPUs.
        old_gpu_indices = [
            gpu_index
            for gpu_index in self.gpu_indices
            # if not self.gpus.supportsGetTotalEnergyConsumption(gpu_index)
        ]
        if old_gpu_indices:
            # Get power_csv_path from env variable
            csv_path = os.getenv("ZEUS_CSV_PATH")
            
            # If the path is not set, error out
            if csv_path is None:
                raise ValueError("ZEUS_CSV_PATH environment variable is not set")

            self.power_monitor = PowerMonitor(
                global_rank=global_rank,
                gpu_indices=old_gpu_indices, 
                update_period=0.05, # In seconds
                csv_path=csv_path, 
            )
        else:
            self.power_monitor = None

    def _get_instant_power(self) -> tuple[dict[int, float], float]:
        """Measure the power consumption of all GPUs at the current time."""
        power_measurement_start_time: float = time()
        power = {
            i: self.gpus.getInstantPowerUsage(i) / 1000.0 for i in self.gpu_indices
        }
        power_measurement_time = time() - power_measurement_start_time
        return power, power_measurement_time

    def begin_window(self, key: str, sync_execution: bool = True) -> None:
        """Begin a new measurement window.

        Args:
            key: Unique name of the measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
        """
        
        # Set environment variable for the power monitor to 1
        self.power_monitor.monitoring_event.set()
        
        # Synchronize execution (e.g., cudaSynchronize) to freeze at the right time.
        if sync_execution and self.gpu_indices:
            sync_execution_fn(self.gpu_indices, sync_with=self.sync_with)
        
        logger.debug("Measurement window '%s' started.", key)
        
        print("Measurement window started")

        
        
    def end_window(
        self, key: str, sync_execution: bool = True, cancel: bool = False
    ) -> Measurement:
        """End a measurement window and return the time and energy consumption.

        Args:
            key: Name of an active measurement window.
            sync_execution: Whether to wait for asynchronously dispatched computations
                to finish before starting the measurement window. For instance, PyTorch
                and JAX will run GPU computations asynchronously, and waiting them to
                finish is necessary to ensure that the measurement window captures all
                and only the computations dispatched within the window.
            cancel: Whether to cancel the measurement window. If `True`, the measurement
                window is assumed to be cancelled and discarded. Thus, an empty Measurement
                object will be returned and the measurement window will not be recorded in
                the log file either. `sync_execution` is still respected.
        """
        
        # Set environment variable for the power monitor to 0
        self.power_monitor.monitoring_event.clear()

        # Synchronize execution (e.g., cudaSynchronize) to freeze at the right time.
        if sync_execution and self.gpu_indices:
            sync_execution_fn(self.gpu_indices, sync_with=self.sync_with)
        
        print("Measurement window ended")
        
        return 0
