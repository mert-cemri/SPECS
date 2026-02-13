#!/bin/bash

# Serve the target model via vLLM OpenAI-compatible API.
# Usage: bash serve_target_model.sh [CUDA_DEVICE] [MODEL] [PORT]

CUDA_DEVICE=${1:-0}
MODEL=${2:-"meta-llama/Llama-3.1-8B-Instruct"}
PORT=${3:-12545}

MODEL_NAME=$(basename "$MODEL")

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

echo "Serving target model: $MODEL on port $PORT (GPU $CUDA_DEVICE)"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "$MODEL_NAME" \
    --tensor-parallel-size 1 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --gpu_memory_utilization 0.45 \
    --max_model_len 8192 \
    --enable_prefix_caching
