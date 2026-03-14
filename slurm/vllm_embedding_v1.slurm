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

mkdir -p logs
module purge
module load rocm/6.3.0
module load singularitypro

MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
PORT=8001
SIF_IMAGE="${HOME}/vllm_rocm_0.6.6.sif"
ROCM_ATTN_PATH="/usr/local/lib/python3.12/dist-packages/vllm/attention/backends/rocm_flash_attn.py"

echo "Starting vLLM Embedding Server (direct file patch)"
echo "Model: ${MODEL_NAME}"
echo "Port: ${PORT}"
echo "Node: $(hostname)"

# Extract rocm_flash_attn.py from container and patch it to handle ENCODER_ONLY
PATCH_DIR="/tmp/vllm_patch_$$"
mkdir -p ${PATCH_DIR}

singularity exec ${SIF_IMAGE} cat ${ROCM_ATTN_PATH} > ${PATCH_DIR}/rocm_flash_attn.py

# Patch: add ENCODER_ONLY case before the else/raise block
# ENCODER_ONLY uses same logic as ENCODER (encoder self-attention)
python3 -c "
import re

with open('${PATCH_DIR}/rocm_flash_attn.py', 'r') as f:
    content = f.read()

# Add 'ENCODER_ONLY' case right before the else clause
old = '''    else:
        raise AttributeError(f\"Invalid attention type {str(attn_type)}\")'''

new = '''    elif attn_type == AttentionType.ENCODER_ONLY:
        # ENCODER_ONLY: like DECODER but non-causal (embedding models)
        # Sequences are in seq_lens (not encoder_seq_lens, which is for enc-dec)
        assert attn_metadata.seq_lens is not None
        assert attn_metadata.seq_lens_tensor is not None

        partial_prefix_sum_eo = 0
        query_seq_start_loc = torch.tensor(
            [0] + [
                partial_prefix_sum_eo := partial_prefix_sum_eo + i
                for i in attn_metadata.seq_lens
            ],
            device=attn_metadata.seq_lens_tensor.device,
            dtype=attn_metadata.seq_lens_tensor.dtype)
        causal_mask = False

        return (query_seq_start_loc, attn_metadata.max_prefill_seq_len,
                query_seq_start_loc, attn_metadata.max_prefill_seq_len,
                attn_metadata.seq_lens, causal_mask)
    else:
        raise AttributeError(f\"Invalid attention type {str(attn_type)}\")'''

if old not in content:
    print('ERROR: Could not find patch target in rocm_flash_attn.py')
    exit(1)

content = content.replace(old, new)

with open('${PATCH_DIR}/rocm_flash_attn.py', 'w') as f:
    f.write(content)

print('PATCH: Successfully added ENCODER_ONLY handling to rocm_flash_attn.py')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed, aborting"
    exit 1
fi

echo "Patch dir: ${PATCH_DIR}"
echo "Bind-mounting patched file over: ${ROCM_ATTN_PATH}"

singularity exec --rocm \
    --bind /cosmos/vast/scratch/$USER:/workspace \
    --bind ${PATCH_DIR}/rocm_flash_attn.py:${ROCM_ATTN_PATH}:ro \
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
