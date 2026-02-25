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

from dataclasses import dataclass
from typing import Literal

from huggingface_hub import get_full_repo_name

from sal.utils.hub import get_dataset_revisions


@dataclass
class Config:
    approach: Literal["specalign_rejection", "speculative_beam_search"] = "specalign_rejection"
    model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    model_ip_address: str = "http://localhost:12549/v1"
    draft_model_path: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    draft_model_ip_address: str = "http://localhost:12549/v1"
    target_model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct"
    target_model_ip_address: str = "http://localhost:12545/v1"
    prm_path: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B"
    prm_ip_address: str = "http://localhost:12550/v1"
    gpu_memory_utilization: float = (
        0.95  # vllm is allocated 0.5 of GPU memory, the PRM uses the rest
    )
    # Output Related Options
    output_dir: str = None
    num_proc: int = None
    push_to_hub: bool = False
    hub_dataset_id: str = None
    overwrite_hub_revision: bool = False
    apply_voting: bool = True

    # Dataset Related Options
    dataset_name: str = "HuggingFaceH4/MATH-500"
    dataset_config: str = None
    dataset_split: str = "test"
    dataset_start: int = None
    dataset_end: int = None
    num_samples: int = None
    logname: str = "unnamed"

    prob_use_draft_model: float = 1

    inspect_dataset: bool = False

    # Chat template related options
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{{}}."
    custom_chat_template: str = None
    # Search Related Options
    n: int = 4
    temperature: float = 0.8
    top_p: float = 1.0
    prm_batch_size: int = 4
    search_batch_size: int = 25
    seed: int = 42
    max_tokens: int = 8192
    target_max_tokens: int = 32768
    agg_strategy: str = "last"  # Options: "last", "min", "prod"

    rejection_threshold: float = 5.0

    switch_back_threshold: float = 0.0

    rm_regularizer: float = 0.1
    # DVTS / Beam Search options
    beam_width: int = 4  # m in the paper
    num_iterations: int = 40
    lookahead: int = 0
    speculative: bool = True
    sample: bool = True
    sample_final_pred: bool = False
    
    period: int = 0
    max_model_len: int = None

    # Beam search options:
    filter_duplicates: bool = True
    sort_completed: bool = False

    def __post_init__(self):
        if self.approach == "speculative_beam_search":
            if self.search_batch_size != 1:
                raise ValueError("search_batch_size should be 1 for speculative_beam_search")

        # Setting up push to hub dataset
        if self.push_to_hub:
            model_name = self.model_path.split("/")[-1]
            if self.hub_dataset_id is None:
                self.hub_dataset_id = get_full_repo_name(
                    f"{model_name}-{self.approach}-prm-completions"
                )
            revisions = get_dataset_revisions(self.hub_dataset_id)

            self.revision = (
                f"{self.dataset_name.replace('/', '_')}--T-{self.temperature}--top_p-{self.top_p}"
                f"--n-{self.n}--m-{self.beam_width}--iters-{self.num_iterations}"
                f"--look-{self.lookahead}--seed-{self.seed}--agg_strategy--{self.agg_strategy}"
            )
            if self.dataset_start is not None and self.dataset_end is not None:
                self.revision = (
                    f"{self.revision}--chunk-{self.dataset_start}_{self.dataset_end}"
                )

            # Early exit if the revision on the Hub already exists
            if not self.overwrite_hub_revision and self.revision in revisions:
                exit()
