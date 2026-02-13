#!/bin/bash

# Serve the draft model via vLLM OpenAI-compatible API.
# Usage: bash serve_draft_model.sh [CUDA_DEVICE] [MODEL] [PORT]

CUDA_DEVICE=${1:-0}
MODEL=${2:-"meta-llama/Llama-3.2-1B-Instruct"}
PORT=${3:-12549}

MODEL_NAME=$(basename "$MODEL")

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Serving draft model: $MODEL on port $PORT (GPU $CUDA_DEVICE)"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enable_prefix_caching \
    --max_model_len 8192 \
    --gpu_memory_utilization 0.40
