#!/bin/bash
#SBATCH --job-name=vllm-embed
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_embed_%j.out
#SBATCH --error=logs/vllm_embed_%j.err

# KnightGPT vLLM Embedding Server for SDSC Cosmos (MI300A)
# Container: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

mkdir -p logs

module purge
module load rocm/6.3.0
module load singularitypro

# Model configuration
MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
PORT=8001
SIF_IMAGE="${HOME}/vllm_rocm_0.6.6.sif"

echo "Starting vLLM Embedding Server"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Node: $(hostname)"

singularity exec --rocm \
    --bind /cosmos/vast/scratch/$USER:/workspace \
    ${SIF_IMAGE} \
    vllm serve "${MODEL_NAME}" \
        --task embedding \
        --trust-remote-code \
        --port ${PORT} \
        --host 0.0.0.0 \
        --enforce-eager \
        --gpu-memory-utilization 0.70 \
        --max-model-len 8192 \
        --swap-space 0 \
        --dtype float16
