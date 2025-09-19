# CharLLM-PPT
Artifact repository of MICRO'25 paper titled "Characterizing the Efficiency of Distributed Training: A Power, Performance, and Thermal Perspective". For more details, please refer to our [paper](https://arxiv.org/abs/2509.10371).

<a name="experiments"></a>
## Experiment Guides

<a name="installation"></a>
### Installing Prerequisites
#### System Requirements:
* Slurm workload manager
* Anaconda 3
* HPC cluster with 32 NVIDIA H200 GPUs and 64 H100 GPUs
* \>1TB storage space

```
  # Create conda environment with name CharLLM-PPT and python version 3.10
  conda create -n CharLLM-PPT python=3.10
  conda activate CharLLM-PPT

  # Navigate to script directory
  export CHARLLM_ROOT=<PATH_TO_THE_PROJECT_ROOT_DIRECTORY>

  # Install prerequisites
  cd CharLLM-PPT/scripts/
  ./install.sh # Expected duration 5 hours
```


<a name="preparing"></a>
### Preparing Scripts and Datasets
Before launching experiments, prepare the scripts for user's cluster environment.
```
  # Before running, update paths and cluster informations in ${CHARLLM_ROOT}/scripts/prepare_scripts.sh
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/
  ./prepare_scripts.sh
```

Once the scripts are ready, download and preprocess Pile dataset.
```
  # Download and preprocess datasets
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/
  ./prepare_datasets.sh # Preprocessed datasets will be saved under ${CHARLLM_ROOT}
```


<a name="launching"></a>
### Launching Experiments

##### Parallelism Sweep (Figure 4)
* Expected duration: 16 hours total (6 hours MP + 10 hours FSDP)
* Results are generated under ${CHARLLM_ROOT}/results/
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/

  # Option 1: Launch both mp and fsdp experiments
  ./parallelism_sweep.sh

  # Option 2: Launch only mp experiments
  ./parallelism_sweep.sh --mp

  # Option 3: Launch only fsdp experiments
  ./parallelism_sweep.sh --fsdp
```

##### H200 Optimization Sweep (Figure 8, 5, 17)
* Expected duration: 60 hours total (24 hours MP + 36 hours FSDP)
* Results are generated under ${CHARLLM_ROOT}/results/
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/

  # Option 1: Launch both mp and fsdp experiments
  ./optimization_sweep_h200.sh

  # Option 2: Launch only mp experiments
  ./optimization_sweep_h200.sh --mp

  # Option 3: Launch only fsdp experiments
  ./optimization_sweep_h200.sh --fsdp
```

##### Microbatch Sweep (Figure 13, 15)
* Expected duration: 48 hours total (24 hours MP + 24 hours FSDP).
* Results are generated under ${CHARLLM_ROOT}/results/
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/

  # Option 1: Launch both mp and fsdp experiments
  ./microbatch_sweep.sh

  # Option 2: Launch only mp experiments
  ./microbatch_sweep.sh --mp

  # Option 3: Launch only fsdp experiments
  ./microbatch_sweep.sh --fsdp
```

##### H100 Optimization Sweep
* Expected duration: 60 hours (24 hours MP + 36 hours FSDP).
* Results are generated under ${CHARLLM_ROOT}/results/
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/

  # Option 1: Launch both mp and fsdp experiments
  ./optimization_sweep_h100.sh

  # Option 2: Launch only mp experiments
  ./optimization_sweep_h100.sh --mp

  # Option 3: Launch only fsdp experiments
  ./optimization_sweep_h100.sh --fsdp
```

<a name="launching"></a>
### Visualizing Results
##### Parallelism Sweep (Figure 4)
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/
  bash ./figure_4.sh # Figures are generated under ${CHARLLM_ROOT}/CharLLM-PPT/figures/
```

##### Optimization Sweep (Figure 8)
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/
  bash ./figure_8.sh
```

##### Microbatch Sweep (Figure 13, 15)
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/
  python ./figure_13.sh # Figures are generated under ${CHARLLM_ROOT}/CharLLM-PPT/figures/
  python ./figure_15.sh
```

##### Heatmaps Sweep (Figure 5, 17)
```
  cd ${CHARLLM_ROOT}/CharLLM-PPT/scripts/visualization/
  ./figure_heatmaps.sh # Figures are generated under ${CHARLLM_ROOT}/CharLLM-PPT/figures/heatmaps and ${CHARLLM_ROOT}/CharLLM-PPT/figures/heatmaps-fsdp
```


<a name="main-changes"></a>
## Main Changes
All changes for CharLLM-PPT are in the following files and marked with `CharLLM-PPT` comment.
* Zeus code is based on https://github.com/ml-energy/zeus (commit 869657a)
* NeMo code is based on https://github.com/NVIDIA/NeMo (commit b70681c)
* Nemo-Framework-Launcher is based on https://github.com/NVIDIA/NeMo-Framework-Launcher (commit 9b715db)
* Megatron-NVIDIA code is based on https://github.com/NVIDIA/Megatron-LM (commit b5d90de)
* Megatron-AMD code is based on https://github.com/ROCm/Megatron-LM (commit f712ab8)
* TransformerEngine code is based on https://github.com/NVIDIA/TransformerEngine (commit 754d2a)

<a name="zeus"></a>
### Zeus 
Zeus is a Python library that monitors GPU metrics using nvml/amd-smi backend. During training, Zeus launches background processes (one per GPU rank) that poll GPU-related metrics.
* `zeus/device/gpu/common.py` (Add additional query functions and their corresponding virtual functions)
* `zeus/device/gpu/nvidia.py` (Add additional wrapper for pynvml functions)
* `zeus/monitor/energy.py` (Add additional metrics and csv paths for class ZeusMonitor)
* `zeus/monitor/power.py` (Add metric collection fuctions for class PowerMonitor)

<a name="nemo"></a>
### NeMo 
NeMo is a distributed machine learning framework designed for NVIDIA GPUs with advanced features.
* `NeMo/nemo/lightning/nemo_logger.py` (Distributed logging for multi-GPU execution)
* `NeMo/nemo/utils/exp_manager.py` (Distributed logging for multi-GPU execution)
* `NeMo/nemo/core/classes/modelPT.py` (Integrate ZeusMonitor)

<a name="megatron-lm"></a>
### Megatron-NVIDIA (NVIDIA/Megatron-LM)
Megatron-NVIDIA is a distributed machine learning framework designed for NVIDIA GPUs.
* `Megatron-NVIDIA/megatron/training/training.py` (Wrap the training loop with ZeusMonitor)

<a name="megatron-amd"></a>
### Megatron-AMD (ROCm/Megatron-LM)
Megatron-AMD is a distributed machine learning framework designed for AMD GPUs.
* `Megatron-AMD/megatron/training/training.py` (Wrap the training loop with ZeusMonitor)

### Submodules
* NeMo-Reference contains old NeMo (9e1ce6f) updated with profiling code.
