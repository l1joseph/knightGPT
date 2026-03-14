#!/bin/bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_infer_%j.out
#SBATCH --error=logs/vllm_infer_%j.err

# KnightGPT vLLM Inference Server for SDSC Cosmos (MI300A)
# Container: rocm/vllm:rocm6.2_mi300_ubuntu20.04_py3.9_vllm_0.6.4

mkdir -p logs

module purge
module load rocm/6.3.0
module load singularitypro

# Load HF_TOKEN from .env if not already set
if [ -z "$HF_TOKEN" ] && [ -f "${HOME}/knightGPT/.env" ]; then
    export HF_TOKEN=$(grep '^HF_TOKEN=' "${HOME}/knightGPT/.env" | cut -d= -f2)
fi

# Model configuration
MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
PORT=8000
TENSOR_PARALLEL=4
SIF_IMAGE="${HOME}/vllm_rocm_0.6.6.sif"

echo "Starting vLLM Inference Server"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "Node: $(hostname)"

singularity exec --rocm \
    --bind /cosmos/vast/scratch/$USER:/workspace \
    --env HF_TOKEN="${HF_TOKEN}" \
    ${SIF_IMAGE} \
    vllm serve "${MODEL_NAME}" \
        --port ${PORT} \
        --host 0.0.0.0 \
        --tensor-parallel-size ${TENSOR_PARALLEL} \
        --enforce-eager \
        --gpu-memory-utilization 0.70 \
        --max-model-len 16384 \
        --swap-space 0 \
        --enable-prefix-caching \
        --dtype float16 \
        --trust-remote-code
