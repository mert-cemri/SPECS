#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set temporary directory to filesystem with more space
import os
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'

# Disable datasets caching globally
import datasets
datasets.disable_caching()

import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.sal.search.specalign_rejection import specalign_rejection
from src.sal.search.beam_search import speculative_beam_search
from src.sal.config import Config
from src.sal.utils.data import get_dataset, save_dataset
from src.sal.utils.parser import H4ArgumentParser
from src.sal.utils.score import score

from external.qwen25_math_evaluation.grader import math_equal
from external.qwen25_math_evaluation.parser import extract_answer, strip_string

import numpy as np
import time
from datasets import Features, Value, Sequence
import wandb

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "specalign_rejection": specalign_rejection,
    "speculative_beam_search": speculative_beam_search,
}

from openai import OpenAI
from transformers import AutoTokenizer


def setup(args):
    # load model
    openai_api_key = "EMPTY"
    draft_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.draft_model_ip_address,
        timeout=1800.0,
    )
    draft_tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    target_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.target_model_ip_address,
        timeout=1800.0,
    )
    target_tokenizer = AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True)
    

    prm_client = OpenAI(
        api_key=openai_api_key,
        base_url=args.prm_ip_address,
        timeout=1800.0,
    )
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_path, trust_remote_code=True)

    return draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer

def get_success(examples, config):
    solutions = examples["pred"]
    gts = examples["answer"]
    success_results = {'success': []}
    for solution, gt in zip(solutions, gts, strict=True):
        pred_ans = extract_answer(solution, data_name=config.dataset_name)
        pred_ans = strip_string(pred_ans, skip_unit=False)
        
        gt_ans = extract_answer(gt, data_name=config.dataset_name)
        gt_ans = strip_string(gt_ans, skip_unit=False)
        success_results['success'].append(math_equal(pred_ans, gt_ans))
    return success_results

def get_success_gpqa(examples, config):
    solutions = examples["pred"]
    gts = examples["answer"]
    success_results = {'success': []}
    for solution, gt in zip(solutions, gts, strict=True):
        equal = (extract_answer(solution, "gpqa", use_last_number=False) == gt)
        success_results['success'].append(equal)
    return success_results

def get_success_amc(examples, config):
    solutions = examples["pred"]
    gts = examples["answer"]
    success_results = {'success': []}
    for solution, gt in zip(solutions, gts, strict=True):
        pred_ans = strip_string(extract_answer(solution, "amc", use_last_number=False), skip_unit=False)
        gt_ans = strip_string(extract_answer(str(gt), "amc", use_last_number=False), skip_unit=False)
        equal = math_equal(pred_ans, gt_ans)
        success_results['success'].append(equal)
    return success_results

def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    # --- Wandb Initialization ---
    # Create a unique run name based on key hyperparameters
    run_name = (
        f"{config.approach}-"
        f"draft_{config.draft_model_path.split('/')[-1]}-"
        f"target_{config.target_model_path.split('/')[-1]}-"
        f"prm_{config.prm_path.split('/')[-1]}-"
        f"reg_{config.rm_regularizer}-"
        f"temp_{config.temperature}-"
        f"n_{config.n}-"
        f"iter_{config.num_iterations}-"
        f"ds_{config.dataset_name.split('/')[-1]}"
    )
    dataset_tag = config.dataset_name.split('/')[-1]


    draft_client, target_client, prm_client, draft_tokenizer, target_tokenizer, prm_tokenizer = setup(config)

    approach_fn = APPROACHES[config.approach]
    dataset = get_dataset(config)

    print(f"Using approach: {config.approach}")

    # Define the features returned by the approach_fn
    # Adjust types (float32/float64, int32/int64) if needed for precision/memory
    if config.approach == "specalign_rejection":
        output_features = Features({
            # Keep existing features from the input dataset implicitly
            **dataset.features,
            # Add or overwrite features returned by approach_fn
            "completions": Sequence(Value("string")),
            "pred": Value("string"),
            "from_draft_model": Sequence(Value("int64")),
            "completion_tokens": Sequence(Value("int64")), # Assuming list of ints per example
            "scores": Sequence(Value("float64")),          # List of floats
            "reward_scores": Sequence(Value("float64")),   # List of floats
            "prob_scores": Sequence(Value("float64")),     # List of floats
            "runtime": Value("float64")                    # Single float
        })
        start = time.time()
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": draft_client, "prm": prm_client, "llm_target": target_client, "draft_tokenizer": draft_tokenizer, "target_tokenizer": target_tokenizer, "prm_tokenizer": prm_tokenizer},
            desc="Running (speculative) search",
            load_from_cache_file=False,
            features=output_features
        )
    else:
        start = time.time()
        dataset = dataset.map(
            approach_fn,
            batched=True,
            batch_size=config.search_batch_size,
            fn_kwargs={"config": config, "llm": draft_client, "prm": prm_client, "llm_target": target_client, "draft_tokenizer": draft_tokenizer, "target_tokenizer": target_tokenizer, "prm_tokenizer": prm_tokenizer},
            desc="Running (speculative) search",
            load_from_cache_file=False
        )


    end = time.time()

    runtime = end - start

    print(f"Runtime: {runtime}")

    dataset = score(dataset, config)

    # Define the features AFTER the score function has potentially modified the dataset
    success_features = Features({
        **dataset.features, # Now includes columns added by score()
        "success": Value("bool") # Or Value("int8") if you prefer 0/1
    })


    if "amc" in config.dataset_name:
        dataset = dataset.map(get_success_amc, batched=True, batch_size=25, fn_kwargs={"config": config}, features=success_features, load_from_cache_file=False)
    else:
        dataset = dataset.map(get_success, batched=True, batch_size=25, fn_kwargs={"config": config}, features=success_features, load_from_cache_file=False)
    success_results = dataset['success']
    success_rate = np.mean(success_results)
    print(f"Success rate: {success_rate}")

    
    save_dataset(dataset, config)


    logger.info("Done 🔥!")
    logger.info(f"Runtime: {runtime}")
    logger.info(f"Success rate: {success_rate}")

if __name__ == "__main__":
    main()
