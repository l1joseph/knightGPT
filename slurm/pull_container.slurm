#!/bin/bash
#SBATCH --job-name=pull-vllm
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=logs/pull_container_%j.out
#SBATCH --error=logs/pull_container_%j.err

# Pull updated vLLM ROCm container for MI300A

mkdir -p logs

module purge
module load singularitypro

export SINGULARITY_TMPDIR=${HOME}/.singularity/tmp
mkdir -p ${SINGULARITY_TMPDIR}

echo "Pulling vLLM ROCm 6.3.1 container..."
echo "This will take 10-30 minutes depending on network speed"
echo "Node: $(hostname)"

singularity pull \
    ${HOME}/vllm_rocm_0.6.6.sif \
    docker://rocm/vllm:rocm6.3.1_mi300_ubuntu22.04_py3.12_vllm_0.6.6

echo "Pull complete. Image saved to ~/vllm_rocm_0.6.6.sif"
ls -lh ${HOME}/vllm_rocm_0.6.6.sif
