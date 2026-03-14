#!/bin/bash
#SBATCH --job-name=knightgpt-ingest
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=logs/weekly_ingest_%j.out
#SBATCH --error=logs/weekly_ingest_%j.err

# KnightGPT Weekly Auto-Ingestion
# Discovers papers from RSS feeds, downloads PDFs, runs pipeline, syncs Neo4j
# Schedule via crontab: 0 0 * * 0 sbatch ~/knightGPT/slurm/weekly_ingest.slurm

mkdir -p logs

module purge
module load rocm/6.3.0
module load singularitypro

SIF_IMAGE="${HOME}/vllm_rocm_0.6.6.sif"
KNIGHTGPT_DIR="${HOME}/knightGPT"
NODE=$(hostname)

echo "=== KnightGPT Weekly Ingestion ==="
echo "Node: ${NODE}"
echo "Date: $(date)"

# Step 1: Start embedding server in background
echo "Starting vLLM embedding server..."
singularity exec --rocm \
    --bind /cosmos/vast/scratch/$USER:/workspace \
    ${SIF_IMAGE} \
    vllm serve "Alibaba-NLP/gte-Qwen2-7B-instruct" \
        --task embedding \
        --trust-remote-code \
        --port 8001 \
        --host 0.0.0.0 \
        --enforce-eager \
        --gpu-memory-utilization 0.70 \
        --max-model-len 8192 \
        --swap-space 0 \
        --dtype float16 &

VLLM_PID=$!
echo "vLLM PID: ${VLLM_PID}"

# Wait for embedding server to be ready
echo "Waiting for embedding server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
        echo "Embedding server ready after ${i}0 seconds"
        break
    fi
    sleep 10
done

# Verify server is up
if ! curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "ERROR: Embedding server failed to start"
    kill ${VLLM_PID} 2>/dev/null
    exit 1
fi

# Step 2: Run auto-ingest with the conda env
cd ${KNIGHTGPT_DIR}

# Update .env with current node's embedding URL
export VLLM_EMBEDDING_URL="http://localhost:8001/v1"

source ~/.bashrc
conda activate knightGPT

python scripts/auto_ingest.py \
    --run-pipeline \
    --sync-neo4j \
    --max-papers 50 \
    --delay 2.0 \
    --log-level INFO

echo "=== Ingestion complete ==="
echo "Date: $(date)"

# Cleanup
kill ${VLLM_PID} 2>/dev/null
wait ${VLLM_PID} 2>/dev/null
