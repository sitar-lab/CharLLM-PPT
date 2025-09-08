#!/bin/bash
#SBATCH --error=${CHARLLM_ROOT}/CharLLM-PPT/results/install/install.err
#SBATCH -C H200
#SBATCH --gpus-per-node=1
#SBATCH --job-name=charllm-install
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --output=${CHARLLM_ROOT}/CharLLM-PPT/results/install/install.out
#SBATCH --time=0-8:00:00


set -euo pipefail

# ---- Config ----
: "${CHARLLM_ROOT:=${HOME}/charllm}"   # export CHARLLM_ROOT before sbatch to override
ENV_NAME="CharLLM-PPT"
CUDA_RUN=cuda_12.4.0_550.54.14_linux.run
CUDA_DIR="${CHARLLM_ROOT}/cuda-12.4"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/${CUDA_RUN}"

echo "[INFO] CHARLLM_ROOT=${CHARLLM_ROOT}"
mkdir -p "${CHARLLM_ROOT}"

# ---- Anaconda env ----
module load anaconda3 || true
# ensure 'conda' works in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Using existing conda env ${ENV_NAME}"
else
  echo "[INFO] Creating conda env ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# ---- CUDA 12.4 local install ----
if [ -x "${CHARLLM_ROOT}/${CUDA_RUN}" ]; then
  echo "[INFO] CUDA installer already downloaded."
else
  echo "[INFO] Downloading CUDA installer ..."
  wget -q -P "${CHARLLM_ROOT}" "${CUDA_URL}"
fi

if [ -d "${CUDA_DIR}" ]; then
  echo "[INFO] CUDA seems installed at ${CUDA_DIR}"
else
  echo "[INFO] Installing CUDA toolkit to ${CUDA_DIR}"
  chmod +x "${CHARLLM_ROOT}/${CUDA_RUN}"
  bash "${CHARLLM_ROOT}/${CUDA_RUN}" --silent --toolkit --toolkitpath="${CUDA_DIR}"
fi

export PATH="${CUDA_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_DIR}/lib64:${LD_LIBRARY_PATH:-}"

# ---- PyTorch + cuDNN ----
python -m pip install --no-user --upgrade pip
python -m pip install --no-user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-user nvidia-cudnn-cu12
python -m pip install --no-user ninja

# Copy cuDNN into local CUDA (headers/libs)
mkdir -p "${CUDA_DIR}/include" "${CUDA_DIR}/lib64"
CUDNN_SITE="$(python -c 'import site,sys; [print(p) for p in site.getsitepackages()]' | grep -m1 .)"
if [ -d "${CUDNN_SITE}/nvidia/cudnn/include" ]; then
  cp -u "${CUDNN_SITE}/nvidia/cudnn/include/"* "${CUDA_DIR}/include" || true
fi
if [ -d "${CUDNN_SITE}/nvidia/cudnn/lib" ]; then
  cp -u "${CUDNN_SITE}/nvidia/cudnn/lib/"* "${CUDA_DIR}/lib64" || true
fi

# ---- Repos layout checks ----
require_dir() {
  [ -d "$1" ] || { echo "[ERROR] Required dir not found: $1"; exit 1; }
}

require_dir "${CHARLLM_ROOT}/CharLLM-PPT"
require_dir "${CHARLLM_ROOT}/CharLLM-PPT/TransformerEngine"
require_dir "${CHARLLM_ROOT}/CharLLM-PPT/Megatron-NVIDIA"
require_dir "${CHARLLM_ROOT}/CharLLM-PPT/NeMo"
require_dir "${CHARLLM_ROOT}/CharLLM-PPT/zeus"

# ---- Apex ----
if [ -d "${CHARLLM_ROOT}/apex/.git" ]; then
  echo "[INFO] Apex already cloned."
else
  git clone https://github.com/NVIDIA/apex.git "${CHARLLM_ROOT}/apex"
fi
cd "${CHARLLM_ROOT}/apex"
git fetch --all
git checkout 312acb || true
python -m pip install --no-user . -v \
  --no-build-isolation \
  --disable-pip-version-check \
  --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"

# ---- FlashAttention ----
if [ -d "${CHARLLM_ROOT}/flash-attention/.git" ]; then
  echo "[INFO] FlashAttention already cloned."
else
  git clone https://github.com/Dao-AILab/flash-attention.git "${CHARLLM_ROOT}/flash-attention"
fi
cd "${CHARLLM_ROOT}/flash-attention"
git fetch --all
git checkout 5639b9 || true
MAX_JOBS=8 python setup.py install

# ---- TransformerEngine ----
cd "${CHARLLM_ROOT}/CharLLM-PPT/TransformerEngine"
export NVTE_FRAMEWORK=pytorch
python -m pip install --no-user .

# ---- pybind11 ----
python -m pip install --no-user pybind11

# ---- Megatron-LM ----
cd "${CHARLLM_ROOT}/CharLLM-PPT/Megatron-NVIDIA"
python -m pip install --no-user .
cd megatron/core/datasets
make -j"$(nproc)"

# ---- NeMo ----
cd "${CHARLLM_ROOT}/CharLLM-PPT/NeMo/requirements"
python -m pip install --no-user -r requirements_common.txt
python -m pip install --no-user -r requirements_nlp.txt
python -m pip install --no-user -r requirements_lightning.txt
cd ..
python -m pip install --no-user .

# ---- Prometheus, Zeus, viz deps ----
python -m pip install --no-user prometheus_client
cd "${CHARLLM_ROOT}/CharLLM-PPT/zeus"
python -m pip install --no-user .
python -m pip install --no-user vistools seaborn pypdf zstandard

echo "[INFO] Installation completed."