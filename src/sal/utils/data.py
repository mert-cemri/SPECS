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

import json
import logging
import time
from pathlib import Path

from datasets import Dataset, load_dataset
from huggingface_hub import (
    create_branch,
    get_full_repo_name,
    list_repo_commits,
    repo_exists,
)
import numpy as np

from sal.config import Config

logger = logging.getLogger()

import os



def get_dataset(config: Config) -> Dataset:
    # Check if the dataset_name points to a local jsonl file
    is_local_jsonl = os.path.exists(config.dataset_name) and config.dataset_name.endswith(".jsonl")

    if is_local_jsonl:
        logger.info(f"Loading local JSONL dataset from: {config.dataset_name}")
        # Load dataset from the local jsonl file
        dataset = load_dataset("json", data_files=config.dataset_name, split="train")

        # Rename columns to match the expected format ("problem", "answer")
        if "question" in dataset.column_names and "problem" not in dataset.column_names:
            dataset = dataset.rename_column("question", "problem")
            logger.info("Renamed column 'question' to 'problem'")
        if "final_answer" in dataset.column_names and "answer" not in dataset.column_names:
            dataset = dataset.rename_column("final_answer", "answer")
            logger.info("Renamed column 'final_answer' to 'answer'")

            # Extract the answer string from the list
            def extract_answer_string(example):
                if isinstance(example["answer"], list) and len(example["answer"]) > 0:
                    example["answer"] = example["answer"][0]
                elif isinstance(example["answer"], list) and len(example["answer"]) == 0:
                     example["answer"] = "" # Handle empty list case
                # Keep as is if it's already a string or other type
                return example

            dataset = dataset.map(extract_answer_string)
            logger.info("Processed 'answer' column to extract string from list.")

        # Verify necessary columns exist after potential renaming

        required_columns = ["problem", "answer"]
        if not all(col in dataset.column_names for col in required_columns):
             logger.warning(f"Loaded dataset missing required columns. Expected: {required_columns}, Found: {dataset.column_names}. Downstream processing might fail.")


    else:
        logger.info(f"Loading dataset from Hugging Face Hub: {config.dataset_name}")
        # Load dataset from Hugging Face Hub (existing behavior)
        dataset = load_dataset(config.dataset_name, name=config.dataset_config, split=config.dataset_split)
        # Ensure MATH dataset also has 'problem' and 'answer' columns if loaded from HF
        # (Usually they do, but good to be explicit if needed)
        if "problem" not in dataset.column_names:
             logger.warning(f"Loaded dataset missing 'problem' column. Found: {dataset.column_names}")
        if "answer" not in dataset.column_names:
             logger.warning(f"Loaded dataset missing 'answer' column. Found: {dataset.column_names}")


    if config.num_samples is not None:
        # Calculate the length of the dataset
        dataset_length = len(dataset)
        logger.info(f"Original dataset size: {dataset_length}")
        
        # Define indices file path
        indices_file = f"selected_indices_seed{42}_n{config.num_samples}_{dataset_length}.json"
        
        # Check if indices file already exists
        if os.path.exists(indices_file):
            # Load existing indices
            with open(indices_file, "r") as f:
                selected_indices = json.load(f)["indices"]
            logger.info(f"Loaded existing indices from {indices_file}")
        else:
            # Create new indices
            rng = np.random.RandomState(42)
            selected_indices = rng.choice(
                dataset_length, 
                size=min(dataset_length, config.num_samples), 
                replace=False
            ).tolist()
            
            # Save the selected indices for reproducibility
            with open(indices_file, "w") as f:
                json.dump({"indices": selected_indices}, f)
            logger.info(f"Created and saved new indices to {indices_file}")
        
        # Use the selected indices to create the dataset subset
        dataset = dataset.select(selected_indices)


    logger.info(f"Final dataset size after slicing/sampling: {len(dataset)}")
    logger.info(f"Dataset columns: {dataset.column_names}")

    # Print some examples from the dataset for inspection
    if config.inspect_dataset:
        if len(dataset) > 0:
            logger.info("Printing sample examples from the dataset:")
            num_examples = min(3, len(dataset))
            for i in range(num_examples):
                example = dataset[i]
                logger.info(f"\nExample {i+1}:")
                
                # Print problem field
                if "problem" in example:
                    problem_preview = example["problem"]
                    # Truncate long problems for readability
                    if len(problem_preview) > 200:
                        problem_preview = problem_preview[:200] + "..."
                    logger.info(f"Problem: {problem_preview}")
                
                # Print answer field
                if "answer" in example:
                    answer_preview = example["answer"]
                    # Truncate long answers for readability
                    if isinstance(answer_preview, str) and len(answer_preview) > 100:
                        answer_preview = answer_preview[:100] + "..."
                    logger.info(f"Answer: {answer_preview}")
                
                # Print other potentially useful fields
                for field in ["idx", "solution", "category", "level"]:
                    if field in example:
                        field_value = example[field]
                        # Truncate long values
                        if isinstance(field_value, str) and len(field_value) > 100:
                            field_value = field_value[:100] + "..."
                        logger.info(f"{field.capitalize()}: {field_value}")
                
                logger.info("-" * 40)
            time.sleep(10)

    return dataset


def save_dataset(dataset, config):
    if config.push_to_hub:
        # Since concurrent pushes can get rejected by the Hub, we make several attempts to push the dataset with try/except
        for _ in range(20):
            try:
                # Create branch from the repo's initial commit.
                # This is needed to avoid branching from a commit on main that already has data
                if repo_exists(config.hub_dataset_id, repo_type="dataset"):
                    initial_commit = list_repo_commits(
                        config.hub_dataset_id, repo_type="dataset"
                    )[-1]
                    create_branch(
                        repo_id=config.hub_dataset_id,
                        branch=config.revision,
                        revision=initial_commit.commit_id,
                        exist_ok=True,
                        repo_type="dataset",
                    )
                url = dataset.push_to_hub(
                    config.hub_dataset_id,
                    revision=config.revision,
                    split="train",
                    private=True,
                    commit_message=f"Add {config.revision}",
                )
                break
            except Exception as e:
                logger.error(f"Error pushing dataset to the Hub: {e}")
                time.sleep(5)
        logger.info(f"Pushed dataset to {url}")
    else:
        if config.output_dir is None:
            if "olympiad" in config.dataset_name:
                config.output_dir = f"results_real_runs_olympiad_new_threshold/{config.logname}"
            elif "amc" in config.dataset_name:
                config.output_dir = f"results_real_runs_amc_new_threshold/{config.logname}"
            else:
                config.output_dir = f"results_real_runs_math500_new_threshold/{config.logname}"

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        model_path = config.model_path.replace("/","")
        if config.logname == "speculative_beam_search":
            target_model_path = config.target_model_path.replace("/","") if config.target_model_path else ""
        else:
            target_model_path = ""
        if config.num_samples is not None:
            num_samples = config.num_samples
        else:
            num_samples = len(dataset)
        dataset.to_json(f"{config.output_dir}/{model_path}-{target_model_path}_n{config.n}_b{config.rm_regularizer}_d_size{num_samples}_t{config.max_tokens}_num_iters{config.num_iterations}_speculative{config.speculative}_temp{config.temperature}_threshold{config.rejection_threshold}_switchback{config.switch_back_threshold}_seed{config.seed}.jsonl", lines=True)
        logger.info(f"Saved completions to {config.output_dir}/{model_path}-{target_model_path}_n{config.n}_b{config.rm_regularizer}_d_size{num_samples}_t{config.max_tokens}_num_iters{config.num_iterations}_speculative{config.speculative}_temp{config.temperature}_threshold{config.rejection_threshold}_switchback{config.switch_back_threshold}_seed{config.seed}.jsonl")


