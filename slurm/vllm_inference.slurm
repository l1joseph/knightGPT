#!/bin/bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_infer_%j.out
#SBATCH --error=logs/vllm_infer_%j.err

# KnightGPT vLLM Inference Server for SDSC Cosmos Cluster
# AMD MI300A APU with ROCm - Using 4 GPUs for tensor parallelism

# Create log directory
mkdir -p logs

# Load required modules
module purge
module load rocm/6.2
module load python/3.11

# Activate conda environment
source ~/.bashrc
conda activate knightgpt

# Set ROCm environment variables for MI300A
export PYTORCH_ROCM_ARCH="gfx942"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1

# Use all 4 GPUs in the node
export HIP_VISIBLE_DEVICES=0,1,2,3

# Model configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
PORT=8000
TENSOR_PARALLEL=4

# Hugging Face cache and token
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
# Set your HF token for gated models
# export HF_TOKEN="your_token_here"

echo "Starting vLLM Inference Server"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "GPUs: ${HIP_VISIBLE_DEVICES}"

# Start vLLM server for inference
# --tensor-parallel-size: Split model across GPUs
# --distributed-executor-backend mp: Use multiprocessing for distribution
# --max-model-len: Maximum context length
# --enable-prefix-caching: Cache common prefixes for efficiency
vllm serve "${MODEL_NAME}" \
    --port ${PORT} \
    --host 0.0.0.0 \
    --tensor-parallel-size ${TENSOR_PARALLEL} \
    --distributed-executor-backend mp \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --enable-prefix-caching \
    --dtype float16 \
    --trust-remote-code

# Note: For production, consider adding:
# --disable-log-requests: Reduce logging overhead
# --compilation-config '{"full_cuda_graph": true}': Enable graph mode for speed
