# Deployment Guide for SDSC Cosmos Cluster

This guide covers deploying KnightGPT on the SDSC Cosmos cluster with AMD MI300A APUs.

## Cluster Overview

**SDSC Cosmos Cluster Specs:**
- 42 nodes with 4 AMD MI300A APUs each
- MI300A: Combined CPU/GPU with 128GB HBM3 per APU
- xGMI interconnect (768 GBps aggregate bandwidth)
- SLURM scheduler
- ROCm software stack

## Prerequisites

### 1. Account Setup
```bash
# Ensure you have access to the GPU partition
sinfo -p gpu

# Check your allocation
sacctmgr show associations user=$USER
```

### 2. Environment Setup
```bash
# Clone repository
git clone https://github.com/l1joseph/knightGPT.git
cd knightGPT

# Load modules
module load rocm/6.2
module load python/3.11

# Create conda environment
conda create -n knightgpt python=3.11 -y
conda activate knightgpt

# Install dependencies
pip install -r requirements.txt
```

### 3. Hugging Face Setup
```bash
# Login to Hugging Face (required for gated models like Llama)
huggingface-cli login

# Set cache directory (use project storage for persistence)
export HF_HOME=/projects/$PROJECT/hf_cache
```

## vLLM Server Deployment

### Embedding Server (1 GPU)

The embedding server uses a single MI300A APU:

```bash
# Submit SLURM job
sbatch slurm/vllm_embedding.slurm

# Or run interactively
srun --partition=gpu --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=4:00:00 --pty bash

# Inside interactive session
module load rocm/6.2
conda activate knightgpt

export PYTORCH_ROCM_ARCH="gfx942"
export VLLM_ROCM_USE_AITER=1

vllm serve Alibaba-NLP/gte-Qwen2-7B-instruct \
    --task embed \
    --trust-remote-code \
    --port 8001 \
    --max-model-len 8192
```

### Inference Server (4 GPUs)

The inference server uses tensor parallelism across 4 APUs:

```bash
# Submit SLURM job
sbatch slurm/vllm_inference.slurm

# Or run interactively
srun --partition=gpu --gres=gpu:4 --cpus-per-task=32 --mem=256G --time=4:00:00 --pty bash

# Inside interactive session
export PYTORCH_ROCM_ARCH="gfx942"
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=1

vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --port 8000 \
    --tensor-parallel-size 4 \
    --distributed-executor-backend mp \
    --max-model-len 32768 \
    --enable-prefix-caching
```

## Model Selection for AMD MI300A

### Recommended Embedding Models

| Model | Size | Context | Notes |
|-------|------|---------|-------|
| Alibaba-NLP/gte-Qwen2-7B-instruct | 7B | 8192 | Best quality |
| BAAI/bge-large-en-v1.5 | 335M | 512 | Fast, efficient |
| intfloat/multilingual-e5-large | 560M | 512 | Multilingual |

### Recommended Inference Models

| Model | Size | GPUs | Notes |
|-------|------|------|-------|
| meta-llama/Llama-3.3-70B-Instruct | 70B | 4 | Recommended |
| Qwen/Qwen2.5-72B-Instruct | 72B | 4 | Alternative |
| meta-llama/Llama-3.1-8B-Instruct | 8B | 1 | Lightweight |

## ROCm Optimization Flags

```bash
# Enable AITER (AI Tensor Engine for ROCm)
export VLLM_ROCM_USE_AITER=1

# Enable AITER MHA (Multi-Head Attention)
export VLLM_ROCM_USE_AITER_MHA=1

# Set architecture for MI300A
export PYTORCH_ROCM_ARCH="gfx942"

# For full CUDA graph mode (better performance)
vllm serve model_name \
    --compilation-config '{"full_cuda_graph": true}'
```

## Data Pipeline

### 1. Upload PDFs
```bash
# Copy PDFs to data directory
cp /path/to/papers/*.pdf data/raw_pdfs/
```

### 2. Run Ingestion Pipeline
```bash
# Interactive job for processing
srun --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2:00:00 --pty bash

# Run pipeline (ensure embedding server is running)
python scripts/ingest_pipeline.py \
    --input data/raw_pdfs/ \
    --output data/processed/ \
    --embedding-url http://localhost:8001/v1
```

### 3. Batch Processing (Large Datasets)
```bash
# Submit batch job
sbatch slurm/batch_ingest.slurm
```

## Web Application Deployment

### On Cosmos (API Server)
```bash
# Run API server
srun --partition=gpu --gres=gpu:0 --cpus-per-task=4 --mem=16G --time=24:00:00 --pty bash

python -m src.api.main --host 0.0.0.0 --port 8080
```

### External Server (Open WebUI)

Deploy Open WebUI on an external server with Cloudflare:

```bash
# On external server
cd docker
docker-compose up -d

# Configure Cloudflare Tunnel
cloudflared tunnel login
cloudflared tunnel create knightgpt
cloudflared tunnel route dns knightgpt knight-lab-dev.org
```

### Cloudflare Configuration

1. Create a Cloudflare Tunnel
2. Configure DNS to point to your server
3. Add the tunnel token to `docker-compose.yaml`

```yaml
# In docker-compose.yaml
cloudflared:
  environment:
    - TUNNEL_TOKEN=your_tunnel_token_here
```

## Monitoring

### Check Server Status
```bash
# Health check
curl http://localhost:8080/health

# vLLM metrics
curl http://localhost:8000/metrics
```

### View Logs
```bash
# SLURM job output
tail -f logs/vllm_infer_*.out

# API logs
tail -f logs/api.log
```

## Troubleshooting

### Common Issues

**Out of Memory:**
```bash
# Reduce GPU memory utilization
vllm serve model --gpu-memory-utilization 0.8

# Use quantization
vllm serve model --quantization fp8
```

**Slow Startup:**
```bash
# Pre-download models
huggingface-cli download Alibaba-NLP/gte-Qwen2-7B-instruct
```

**Connection Refused:**
```bash
# Check if server is running
ss -tlnp | grep 8000

# Check firewall
sudo iptables -L
```

## Resource Recommendations

| Component | CPUs | Memory | GPUs | Time |
|-----------|------|--------|------|------|
| Embedding Server | 16 | 64GB | 1 | 24h |
| Inference Server | 32 | 256GB | 4 | 24h |
| API Server | 4 | 16GB | 0 | 24h |
| Batch Ingestion | 8 | 32GB | 1 | 4h |

## Security Notes

- Use environment variables for sensitive data
- Enable authentication on Open WebUI
- Restrict API access to internal networks
- Use HTTPS via Cloudflare for external access
