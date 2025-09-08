# Download https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run to CHARLLM_ROOT
wget -P ${CHARLLM_ROOT} https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

# Install CUDA to CHARLLM_ROOT
chmod +x ${CHARLLM_ROOT}/cuda_12.4.0_550.54.14_linux.run
bash ${CHARLLM_ROOT}/cuda_12.4.0_550.54.14_linux.run --silent --toolkit --toolkitpath=${CHARLLM_ROOT}/cuda-12.4

# export PATH and LD_LIBRARY_PATH
export PATH=${CHARLLM_ROOT}/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=${CHARLLM_ROOT}/cuda-12.4/lib64:$LD_LIBRARY_PATH

# Create conda environment CharLLM-PPT with python 3.10
# module load anaconda3
# conda create -n CharLLM-PPT python=3.10 -y

# Install PyTorch with CUDA 12.4
python -m pip install --no-user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install --no-user nvidia-cudnn-cu12
python -m pip install ninja

# Copy cudnn files from conda environment to CHARLLM_ROOT
cp ~/.conda/envs/CharLLM-PPT/lib/python3.10/site-packages/nvidia/cudnn/include/* ${CHARLLM_ROOT}/cuda-12.4/include
cp ~/.conda/envs/CharLLM-PPT/lib/python3.10/site-packages/nvidia/cudnn/lib/* ${CHARLLM_ROOT}/cuda-12.4/lib64

# Clone apex under CHARLLM_ROOT
git clone https://github.com/NVIDIA/apex.git ${CHARLLM_ROOT}/apex

# Checkout apex to 312acb
cd ${CHARLLM_ROOT}/apex
git checkout 312acb

# Install Apex with global options
python -m pip install --no-user . -v \
--no-build-isolation \
--disable-pip-version-check \
--no-cache-dir \
--config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm" || { echo "Apex installation failed"; exit 1; }

cd ../

# FlashAttention
git clone https://github.com/Dao-AILab/flash-attention.git ${CHARLLM_ROOT}/flash-attention
cd ${CHARLLM_ROOT}/flash-attention
git checkout 5639b9

# Install FlashAttention (change MAX_JOBS based on your environment)
MAX_JOBS=8 python setup.py install || { echo "FlashAttention installation failed"; exit 1; }

cd ../

# TransformerEngine
cd ${CHARLLM_ROOT}/CharLLM-PPT/TransformerEngine/

# Install TransformerEngine
export NVTE_FRAMEWORK=pytorch
python -m pip install --no-user . || { echo "TransformerEngine installation failed"; exit 1; }

cd ../

# Install pybind11
python -m pip install --no-user pybind11

# Install Megatron-LM
cd ${CHARLLM_ROOT}/CharLLM-PPT/Megatron-NVIDIA/
python -m pip install --no-user . || { echo "Megatron-LM installation failed"; exit 1; }
cd megatron/core/datasets
make

# cd ../

# Install required packages for NeMo
cd ${CHARLLM_ROOT}/CharLLM-PPT/NeMo/requirements/
python -m pip install --no-user -r ./requirements_common.txt || { echo "NeMo common requirements installation failed"; exit 1; }
python -m pip install --no-user -r ./requirements_nlp.txt || { echo "NeMo NLP requirements installation failed"; exit 1; }
python -m pip install --no-user -r ./requirements_lightning.txt || { echo "NeMo Lightning requirements installation failed"; exit 1; }

cd ../

# Install NeMo
python -m pip install --no-user . || { echo "NeMo installation failed"; exit 1; }

# Install Prometheus Python Client
python -m pip install --no-user prometheus_client

# Install Zeus
cd ${CHARLLM_ROOT}/CharLLM-PPT/zeus/
python -m pip install --no-user .

# Install visualization dependencies
python -m pip install vistools seaborn pypdf zstandard