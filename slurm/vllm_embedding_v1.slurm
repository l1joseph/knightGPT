#!/bin/bash
#SBATCH --job-name=vllm-embed
#SBATCH --partition=cluster        # Cosmos partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_embed_%j.out
#SBATCH --error=logs/vllm_embed_%j.err

# KnightGPT vLLM Embedding Server for SDSC Cosmos Cluster
# AMD MI300A APU with ROCm

# Create log directory
mkdir -p logs

# Purge any loaded modules and load ROCm + Python
module purge
module load rocm/6.3.0
module load amd/6.3.0

# Activate conda environment
source ~/.bashrc
conda activate knightGPT

# ROCm environment variables
export PYTORCH_ROCM_ARCH="gfx942"
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Model configuration
MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
PORT=8001

# Hugging Face cache directory
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

echo "Starting vLLM Embedding Server"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "GPU: ${HIP_VISIBLE_DEVICES}"
echo "ROCm Arch: ${PYTORCH_ROCM_ARCH}"

# Start vLLM embedding server
vllm serve "${MODEL_NAME}" \
    --task embed \
    --trust-remote-code \
    --port ${PORT} \
    --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.8 \
    --dtype float16

