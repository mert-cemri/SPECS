#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install -e "$SCRIPT_DIR"

pip install -e "$SCRIPT_DIR/external/skywork_o1_prm_inference"

pip install -e "$SCRIPT_DIR/external/qwen25_math_evaluation"
