#!/bin/bash

# Serve the Process Reward Model (PRM) via vLLM OpenAI-compatible API.
# Usage: bash serve_prm.sh [CUDA_DEVICE] [MODEL] [PORT]

CUDA_DEVICE=${1:-0}
MODEL=${2:-"Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"}
PORT=${3:-12550}

MODEL_NAME=$(basename "$MODEL")

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Serving PRM: $MODEL on port $PORT (GPU $CUDA_DEVICE)"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len 8192 \
    --enforce-eager \
    --enable_prefix_caching \
    --gpu_memory_utilization 0.95
