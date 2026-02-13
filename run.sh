#!/bin/bash

# Example: run SpecAlign rejection sampling on AMC23 dataset
# Adjust ports to match your serve scripts and models to match your setup.

for n in 4 8 16; do
for seed in 42 43 45; do
  python scripts/test_time_compute.py recipes/specalign_rejection_amc.yaml \
    --model_path="meta-llama/Llama-3.2-1B-Instruct" \
    --target_model_path="meta-llama/Llama-3.1-8B-Instruct" \
    --prm_path="Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B" \
    --draft_model_path="meta-llama/Llama-3.2-1B-Instruct" \
    --draft_model_ip_address="http://localhost:12549/v1" \
    --target_model_ip_address="http://localhost:12545/v1" \
    --prm_ip_address="http://localhost:12550/v1" \
    --rm_regularizer=10 \
    --temperature=0.3 \
    --n=$n \
    --max_tokens=512 \
    --seed=$seed \
    --num_iterations=30 \
    --rejection_threshold=7 \
    --switch_back_threshold=0.7
  done
done
