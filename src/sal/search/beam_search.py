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
import copy
import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams

from .utils import Beam, build_conv, generate_k_steps_and_score, prepare_input, derive_step_rewards_vllm

logger = logging.getLogger()
from sal.utils.score import aggregate_scores

import torch

import time

def __speculative_beam_search(batch_of_prompts, config, llm, prm, llm_target = None, draft_tokenizer = None, target_tokenizer=None, prm_tokenizer=None) -> list[Beam]:

    if llm_target is None:
        llm_target = llm

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
    )

    beams: list[Beam] = []
    for prompt in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=prompt,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,  # New flag to track completion
                    stop_reasons=None,
                    best_scores=[],
                    all_scores=[],
                    history=[],
                    reward_scores=[],
                    prob_scores=[],
                    final_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams: list[Beam] = []

    for i in tqdm(range(config.num_iterations), desc="Speculative beam search iterations"):
        if i == 0:
            active_beams = [b for b in beams if not b.pruned]
        else:
            active_beams = [b for b in active_beams if not b.pruned]

        # Duplicate active beams to ensure that we have config.n beams per iteration
        if len(active_beams) != config.n:
            repeats = (config.n // len(active_beams)) + 1
            logger.debug(
                f"Extending active_beams with {repeats} repetitions to reach size {config.n}"
            )
            extended_active_beams = [
                copy.deepcopy(b) for b in (active_beams * repeats)[: config.n]
            ]
            active_beams = extended_active_beams
            if len(active_beams) != config.n:
                raise ValueError(
                    f"Expected {config.n} active beams, but got {len(active_beams)}"
                )

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_beams
        ]

        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = draft_tokenizer
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        lookahead = 0 if i == config.num_iterations - 1 else config.lookahead

        gen_results = generate_k_steps_and_score(
            templated_convs, lookahead, llm, sampling_params, beam_width=1, llm_target=llm_target, target_tokenizer=target_tokenizer, config = config
        )

        prompts = []
        completions = []
        agg_scores = []
        for beam, gen_result in zip(active_beams, gen_results, strict=True):
            beam.next_texts = gen_result.next_texts
            beam.stop_reasons = gen_result.stop_reasons
            beam.lookahead_texts = gen_result.lookahead_texts
            beam.completion_tokens += gen_result.completion_tokens
            beam.prob_scores = gen_result.prob_scores
            beam.current_text += beam.next_texts[0]
            beam.history.append(beam.next_texts[0])

            if (
                beam.stop_reasons[0] == "EOS"
                or beam.next_texts[0] == ""
                or beam.stop_reasons[0] == "length"
            ):
                beam.completed = True
                completed_beams.append(beam)

            prompts.append(beam.prompt)
            completions.append(beam.current_text)

        processed_data = [
            prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token="\n\n")
            for p, full_resp in zip(prompts, completions)
        ]
        input_ids, steps, reward_flags = zip(*processed_data)
        prm_start_time = time.time()
        rewards = prm.embeddings.create(
            input=input_ids,
            model=config.prm_path.split("/")[-1],
        )
        prm_end_time = time.time()
        prm_time = prm_end_time - prm_start_time
        prm_scores = derive_step_rewards_vllm(rewards, reward_flags)

        for beam, step_rewards in zip(active_beams, prm_scores, strict=True):
            beam.all_scores = [step_rewards]
            beam.reward_scores = [step_rewards[-1]]
            beam.final_scores = [beam.prob_scores[0] + config.rm_regularizer*beam.reward_scores[0]]

        # Now filter active_beams and agg_scores for beams that are completed
        agg_scores = [
            b.final_scores for b in active_beams if not b.completed
        ]
        active_beams = [b for b in active_beams if not b.completed]

        if len(active_beams) == 0:
            break

        if config.filter_duplicates:
            unique_beam_dict = {}
            for i, b in enumerate(active_beams):
                if b.current_text not in unique_beam_dict:
                    unique_beam_dict[b.current_text] = i
            active_beams = [active_beams[i] for i in unique_beam_dict.values()]
            agg_scores = [agg_scores[i] for i in unique_beam_dict.values()]
        agg_scores = torch.softmax(torch.tensor(agg_scores).flatten(), dim=0).numpy()

        if config.sample:
            top_indices = np.random.choice(np.arange(len(agg_scores)), size=min(config.n//config.beam_width, len(agg_scores)), p=agg_scores, replace=False)
        else:
            top_indices = np.argsort(agg_scores)[
                -min(config.n // config.beam_width, len(agg_scores)):
            ]

        for idx, beam in enumerate(active_beams):
            if idx not in top_indices:
                beam.pruned = True

    if config.sort_completed:
        completed_beams = sorted(completed_beams, key=lambda x: x.final_scores[0], reverse=True)[:config.n]
    else:
        completed_beams = completed_beams[:config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams


def speculative_beam_search(examples, config, llm=None, prm=None, llm_target=None, draft_tokenizer=None, target_tokenizer=None, prm_tokenizer=None):
        problems = examples["problem"]
        start_time = time.time()
        beam_results = __speculative_beam_search(problems, config, llm, prm, llm_target, draft_tokenizer, target_tokenizer, prm_tokenizer)
        end_time = time.time()
        time_taken = end_time - start_time

        # Group together alike beams and store in the dataset
        grouped_results = defaultdict(list)
        for results in beam_results:
            grouped_results[results.prompt].append(results)

        results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "reward_scores": [], "prob_scores": [], "runtime": [time_taken] * len(problems)}

        for p in problems:
            beams = grouped_results[p]
            completions = [b.current_text for b in beams]
            agg_scores = [
                b.final_scores[0] for b in beams
            ]
            if config.sample_final_pred:
                pred = completions[np.random.choice(np.arange(len(agg_scores)), size=1, p=agg_scores)]
            else:
                pred = completions[np.argmax(agg_scores)]
            results["completions"].append(completions)
            results["scores"].append(agg_scores)
            results["reward_scores"].append([b.reward_scores[0] for b in beams])
            results["prob_scores"].append([b.prob_scores[0] for b in beams])
            results["pred"].append(pred)
            results["completion_tokens"].append([b.completion_tokens for b in beams])
        return results
