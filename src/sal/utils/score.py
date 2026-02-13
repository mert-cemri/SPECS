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


from datasets import Dataset
from tqdm import tqdm

from sal.config import Config
from sal.utils.math import (
    aggregate_scores,
    compute_maj_pred,
    compute_naive_pred,
    compute_weighted_pred,
    extract_completion_answers,
    subsample_completions,
)


def score(dataset: Dataset, config: Config) -> Dataset:
    # Use fewer processes to reduce temporary file creation
    num_proc = min(config.num_proc, 2) if hasattr(config, 'num_proc') and config.num_proc else 1
    
    dataset = dataset.map(
        lambda x: {"agg_scores": x["scores"]},
        load_from_cache_file=False,  # Disable caching
        num_proc=num_proc
    )
    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing majority & weighted predictions"):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Subsample {n}",
            load_from_cache_file=False  # Disable caching
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Extract answers {n}",
            load_from_cache_file=False  # Disable caching
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Compute weighted pred {n}",
            load_from_cache_file=False  # Disable caching
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Compute majority pred {n}",
            load_from_cache_file=False  # Disable caching
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=num_proc,
            desc=f"Compute naive pred {n}",
            load_from_cache_file=False  # Disable caching
        )
        # Nuke unused columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset
