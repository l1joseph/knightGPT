#!/bin/bash
#SBATCH --job-name=vllm-infer
#SBATCH --partition=cluster       # Cosmos partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vllm_infer_%j.out
#SBATCH --error=logs/vllm_infer_%j.err

# KnightGPT vLLM Inference Server for SDSC Cosmos Cluster
# AMD MI300A APU with ROCm - Using 4 GPUs for tensor parallelism
# Runs inside Singularity container with vLLM ROCm image

# Create log directory
mkdir -p logs

# Purge any loaded modules and load ROCm
module purge
module load rocm/6.3.0
module load amd/6.3.0

# ROCm environment variables
export PYTORCH_ROCM_ARCH="gfx942"
export HIP_VISIBLE_DEVICES=0,1,2,3
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Enable vLLM ROCm-specific optimizations
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1

# Threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Model configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
PORT=8000
TENSOR_PARALLEL=4

# Hugging Face cache
export HF_HOME="${HOME}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"

# Singularity container image
SIF_IMAGE="${HOME}/vllm_rocm.sif"

# Apptainer temp directories on scratch
export APPTAINER_TMPDIR=/ddn_scratch/l1joseph/.apptainer/tmp
export APPTAINER_CACHEDIR=/ddn_scratch/l1joseph/.apptainer/cache
mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

echo "Starting vLLM Inference Server (Singularity)"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Tensor Parallel: ${TENSOR_PARALLEL}"
echo "GPUs: ${HIP_VISIBLE_DEVICES}"
echo "Container: ${SIF_IMAGE}"
echo "Node: $(hostname)"

# Start vLLM inference server inside Singularity container
singularity exec --nv \
    --bind /ddn_scratch:/ddn_scratch \
    --bind ${HF_HOME}:${HF_HOME} \
    ${SIF_IMAGE} \
    vllm serve "${MODEL_NAME}" \
        --port ${PORT} \
        --host 0.0.0.0 \
        --tensor-parallel-size ${TENSOR_PARALLEL} \
        --distributed-executor-backend mp \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.70 \
        --enforce-eager \
        --enable-prefix-caching \
        --dtype float16 \
        --trust-remote-code
