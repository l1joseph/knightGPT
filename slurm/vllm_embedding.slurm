#!/bin/bash
#SBATCH --job-name=vllm-embed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_embed_%j.out
#SBATCH --error=logs/vllm_embed_%j.err

# KnightGPT vLLM Embedding Server for SDSC Cosmos Cluster
# AMD MI300A APU with ROCm

# Create log directory
mkdir -p logs

# Load required modules (adjust based on Cosmos module system)
module purge
module load rocm/6.2
module load python/3.11

# Activate conda environment
source ~/.bashrc
conda activate knightgpt

# Set ROCm environment variables for MI300A
export PYTORCH_ROCM_ARCH="gfx942"
export VLLM_ROCM_USE_AITER=1
export HIP_VISIBLE_DEVICES=0

# Model configuration
MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
PORT=8001

# Hugging Face cache directory (use project storage)
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

echo "Starting vLLM Embedding Server"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "GPU: ${HIP_VISIBLE_DEVICES}"
echo "ROCm Arch: ${PYTORCH_ROCM_ARCH}"

# Start vLLM server for embeddings
# --task embed: Run in embedding mode
# --trust-remote-code: Required for some models
# --max-model-len: Maximum sequence length
vllm serve "${MODEL_NAME}" \
    --task embed \
    --trust-remote-code \
    --port ${PORT} \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --dtype float16

# The server will run until the job is killed or time limit reached
