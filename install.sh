#!/bin/bash

pip install -e .

# Install the skywork_o1_prm_inference package
cd external/skywork_o1_prm_inference
pip install -e .

# Install the qwen25_math_evaluation package
cd ../qwen25_math_evaluation
pip install -e .
