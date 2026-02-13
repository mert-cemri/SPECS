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
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import SamplingParams

from sal.utils.score import aggregate_scores
from .utils import Beam, build_conv, generate_k_steps_and_score

import torch
import wandb

from ..utils.eval_utils import extract_answer, strip_string, math_equal

logger = logging.getLogger(__name__)


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


def __speculative_beam_search(batch_of_prompts, config, llm, prm, llm_target=None, draft_tokenizer=None, target_tokenizer=None, prm_tokenizer=None) -> list[Beam]:

    from_draft_model = []
    config.beam_width = config.n

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
                    completed=False,
                    stop_reasons=None,
                    best_scores=[],
                    all_scores=[],
                    history=[],
                    reward_scores=[],
                    prob_scores=[],
                    final_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                    from_draft_model=[],
                )
            )

    completed_beams: list[Beam] = []

    previous_found_good = False
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

        copy_of_active_beams = copy.deepcopy(active_beams)
        if previous_found_good:  # Try using draft model output first
            gen_results = generate_k_steps_and_score(
                templated_convs, lookahead, llm, sampling_params, beam_width=1,
                llm_target=llm_target, target_tokenizer=target_tokenizer,
                prm=prm, prm_tokenizer=prm_tokenizer,
                problem=active_beams[0].prompt, config=config
            )

            prompts = []
            completions = []
            prm_scores = []

            for beam, gen_result in zip(active_beams, gen_results, strict=True):
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                beam.completion_tokens += gen_result.completion_tokens
                beam.prob_scores = gen_result.prob_scores
                beam.current_text += beam.next_texts[0]
                beam.history.append(beam.next_texts[0])
                beam.reward_scores = gen_result.reward_scores
                prm_scores.append(gen_result.reward_scores)

                if (
                    beam.stop_reasons[0] == "EOS"
                    or beam.next_texts[0] == ""
                    or beam.stop_reasons[0] == "length"
                ):
                    beam.completed = True
                    completed_beams.append(beam)

                prompts.append(beam.prompt)
                completions.append(beam.current_text)

            for beam, step_rewards in zip(active_beams, prm_scores, strict=True):
                beam.all_scores = [step_rewards[-1]]
                beam.reward_scores = [step_rewards[-1][-1]]
                beam.final_scores = [beam.prob_scores[0] + config.rm_regularizer * beam.reward_scores[0]]

            # Filter active_beams and agg_scores for beams that are completed
            agg_scores = [
                b.final_scores[0] for b in active_beams if not b.completed
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

            found_good = False
            for i, beam in enumerate(active_beams):
                if beam.final_scores[0] > config.rejection_threshold:
                    found_good = True
                else:
                    beam.pruned = True

            if found_good:
                non_pruned_dict = {}
                for i, beam in enumerate(active_beams):
                    if not beam.pruned:
                        non_pruned_dict[beam.current_text] = i

                active_beams = [b for b in active_beams if not b.pruned]
                agg_scores = [agg_scores[i] for i in non_pruned_dict.values()]
                agg_scores = torch.softmax(torch.tensor(agg_scores).flatten() / config.temperature, dim=0).numpy()

                try_again_flag = False
                try:
                    top_indices = np.random.choice(np.arange(len(agg_scores)), size=min(config.n // config.beam_width, len(agg_scores)), p=agg_scores, replace=False)
                except Exception as e:
                    logger.warning(f"Softmax sampling failed: {e}, agg_scores={agg_scores}")
                    try_again_flag = True
                if try_again_flag:
                    try:
                        top_indices = np.random.choice(np.arange(len(agg_scores)), size=len(agg_scores), p=agg_scores, replace=False)
                    except Exception:
                        top_indices = np.argsort(agg_scores)[
                            -min(config.n // config.beam_width, len(agg_scores)):
                        ]
                for idx, beam in enumerate(active_beams):
                    if idx not in top_indices:
                        beam.pruned = True

                previous_found_good = True
            else:
                previous_found_good = False

        else:
            found_good = False

        logger.debug(f"found_good: {found_good}")

        if found_good:
            from_draft_model.append(1)
        else:
            from_draft_model.append(0)

        if not found_good:
            # Fallback: sample from the target model
            active_beams = copy_of_active_beams
            gen_results = generate_k_steps_and_score(
                templated_convs, lookahead, llm, sampling_params, beam_width=1,
                llm_target=llm_target, target_tokenizer=target_tokenizer,
                prm=prm, prm_tokenizer=prm_tokenizer,
                problem=active_beams[0].prompt, config=config, use_target_as_draft=True
            )

            prompts = []
            completions = []
            prm_scores = []
            for beam, gen_result in zip(active_beams, gen_results, strict=True):
                beam.next_texts = gen_result.next_texts
                beam.stop_reasons = gen_result.stop_reasons
                beam.lookahead_texts = gen_result.lookahead_texts
                beam.completion_tokens += gen_result.completion_tokens
                beam.prob_scores = gen_result.prob_scores
                beam.current_text += beam.next_texts[0]
                beam.history.append(beam.next_texts[0])
                beam.reward_scores = gen_result.reward_scores
                prm_scores.append(gen_result.reward_scores)

                if (
                    beam.stop_reasons[0] == "EOS"
                    or beam.next_texts[0] == ""
                    or beam.stop_reasons[0] == "length"
                ):
                    beam.completed = True
                    completed_beams.append(beam)

                prompts.append(beam.prompt)
                completions.append(beam.current_text)

            for beam, step_rewards in zip(active_beams, prm_scores, strict=True):
                beam.all_scores = [step_rewards[-1]]
                beam.reward_scores = [step_rewards[-1][-1]]
                # When using target model, score by reward only (no prob_scores term)
                beam.final_scores = [beam.reward_scores[0]]

            # Filter active_beams and agg_scores for beams that are completed
            agg_scores = [
                b.final_scores[0] for b in active_beams if not b.completed
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

            for i, beam in enumerate(active_beams):
                if beam.final_scores[0] > config.switch_back_threshold:
                    found_good = True

            if found_good:
                previous_found_good = True
            else:
                previous_found_good = False

            agg_scores = np.array(agg_scores).flatten()
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
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    for beam in completed_beams:
        beam.from_draft_model = from_draft_model

    return completed_beams


from ..utils.eval_utils import get_success_olympiad


def specalign_rejection(examples, config, llm=None, prm=None, llm_target=None, draft_tokenizer=None, target_tokenizer=None, prm_tokenizer=None):
        problems = examples["problem"]
        batch_size = len(problems)
        start_time = time.time()
        try:
            beam_results = __speculative_beam_search(problems, config, llm, prm, llm_target, draft_tokenizer, target_tokenizer, prm_tokenizer)
        except Exception as e:
            logger.error(f"Beam search failed: {e}")
            beam_results = [
                Beam(
                    prompt=problems[0],
                    index=0,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,
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
            ]
            empty_results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "reward_scores": [], "prob_scores": [], "runtime": [0] * len(problems), "from_draft_model": []}
            for p in problems:
                completions = ['no solution']
                agg_scores = [0]
                if config.sample_final_pred:
                    try:
                        pred = completions[np.random.choice(np.arange(len(agg_scores)), size=1, p=agg_scores)]
                    except Exception as e:
                        logger.warning(f"Sampling failed (sum={sum(agg_scores)}): {e}")
                        pred = completions[np.argmax(agg_scores)]
                else:
                    pred = completions[np.argmax(agg_scores)]
                empty_results["completions"].append(completions)
                empty_results["scores"].append(agg_scores)
                empty_results["reward_scores"].append([0])
                empty_results["prob_scores"].append([0])
                empty_results["pred"].append(pred)
                empty_results["completion_tokens"].append([0])
                empty_results["from_draft_model"].append([0])

            return empty_results

        end_time = time.time()
        time_taken = end_time - start_time

        # Group together alike beams and store in the dataset
        grouped_results = defaultdict(list)
        for results in beam_results:
            grouped_results[results.prompt].append(results)

        results = {"completions": [], "pred": [], "completion_tokens": [], "scores": [], "reward_scores": [], "prob_scores": [], "runtime": [time_taken] * batch_size, "from_draft_model": []}
        for p in problems:
            beams = grouped_results[p]
            completions = [b.current_text for b in beams]
            agg_scores = [
                b.final_scores[0] for b in beams
            ]
            pred = completions[np.argmax(np.array(agg_scores).flatten())]
            results["completions"].append(completions)
            results["reward_scores"].append([b.reward_scores[0] if b.reward_scores else 0.0 for b in beams])
            results["scores"].append(agg_scores)
            results["prob_scores"].append([b.prob_scores[0] if b.prob_scores else 0.0 for b in beams])
            results["pred"].append(pred)
            results["completion_tokens"].append([b.completion_tokens for b in beams])

            results["from_draft_model"].append(beams[0].from_draft_model)

            try:
                if "amc" in config.dataset_name:
                    pred_ans = extract_answer(pred)
                    pred_ans = strip_string(pred_ans, skip_unit=False)
                    gt_ans = examples["answer"][0]
                    gt_ans = str(gt_ans)
                    isequal = math_equal(pred_ans, gt_ans)
                else:
                    pred_ans = extract_answer(pred)
                    pred_ans = strip_string(pred_ans, skip_unit=False)
                    isequal = math_equal(pred_ans, examples["answer"][0])
                if isequal:
                    success = 1
                else:
                    success = 0
            except Exception:
                success = 0
                logger.warning("Success could not be computed")

        # Log batch summary statistics to wandb
        if wandb.run:
            wandb.log({
                "batch/runtime": time_taken,
                "batch/success": success
            })

        return results
