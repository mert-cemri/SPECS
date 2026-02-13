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
from dataclasses import dataclass, field

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from vllm import LLM, SamplingParams

logger = logging.getLogger()
import torch

import time

def build_conv(
    prompt: str, response: str | None, system_prompt: str
) -> list[dict[str, str]]:
    conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    if response != "":
        conversation.append({"role": "assistant", "content": response})

    return conversation


def last(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return x[-1]


def list_mean(x):
    if len(x) == 0:
        logger.warning("empty list")
        return 0
    return np.mean(x)

def derive_step_rewards(rewards, reward_flags):
    batch_size = rewards.shape[0]
    batch_step_rewards = []
    for i in range(batch_size):
        rewards_indices = torch.nonzero(reward_flags[i] == 1).view(-1)
        step_rewards = [rewards[i][rewards_indices[j]].item() for j in range(len(rewards_indices))]
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards

def sigmoid(x):
    return 1/(np.exp(-x) + 1)
    
def derive_step_rewards_vllm(raw_rewards, batch_reward_flags):
    batch_step_rewards = []
    for idx,data in enumerate(raw_rewards.data):
        rewards = data.embedding
        reward_flags = batch_reward_flags[idx]

        step_rewards = [sigmoid(reward) for reward,flag in zip(rewards,reward_flags) if flag == 1]   
        batch_step_rewards.append(step_rewards)
    return batch_step_rewards

def prepare_input(problem, response, tokenizer, step_token):
    prompt_ids = tokenizer.encode(tokenizer.bos_token + problem + "\n")
    response_ids = []
    steps = []
    reward_flags = [0] * len(prompt_ids)
    step_token_id = tokenizer.encode(step_token)[-1]
    for idx, step in enumerate(response.split(step_token)):
        if step != "":
            step_ids = tokenizer.encode(step)
        else:
            step_ids = []
        step_ids += [step_token_id]
        step = step + step_token
        flag = [0] * len(step_ids)
        flag[-1] = 1
        response_ids.extend(step_ids)
        reward_flags.extend(flag)
        steps.append(step)
    input_ids = prompt_ids + response_ids
    return input_ids, steps, reward_flags

def run_target_model(prompts, target_client, args, target_tokenizer, max_context_length=8192, use_target_as_draft=False):
    """Run the target model and return results, truncating inputs if necessary."""
    start_time = time.time()

    truncated_prompts = []
    original_indices = [] # Keep track of original indices if needed later, though sorting handles it here
    for i, prompt in enumerate(prompts):
        input_ids = target_tokenizer.encode(prompt)
        if len(input_ids) > max_context_length:
            logger.warning(
                f"Target model input {i} length ({len(input_ids)}) exceeds max context length ({max_context_length}). Truncating from the beginning."
            )
            # Keep the end of the input
            truncated_input_ids = input_ids[-max_context_length:]
            # Decode back to string, skipping special tokens if necessary depending on tokenizer behavior
            truncated_prompt = target_tokenizer.decode(truncated_input_ids, skip_special_tokens=True) # Adjust skip_special_tokens if needed
            truncated_prompts.append(truncated_prompt)
        else:
            truncated_prompts.append(prompt)
        original_indices.append(i) # Store original index

    # Pass the potentially truncated prompts to the API
    if use_target_as_draft:
        target_logits_unordered = target_client.completions.create(
                    model=args.draft_model_path.split("/")[-1],
                    prompt=truncated_prompts, # Use truncated prompts
                    # generated_tokens = last_step,
                    echo=True,
                    logprobs=1, #K
                    n=1,
                    max_tokens=0
                ).choices

    else:    
        target_logits_unordered = target_client.completions.create(
            model=args.target_model_path.split("/")[-1],
            prompt=truncated_prompts, # Use truncated prompts
            # generated_tokens = last_step,
            echo=True,
            logprobs=1, #K
            n=1,
            max_tokens=0
        ).choices

    # # Re-sort based on the original index stored in the response 'index' field
    target_logits = sorted(target_logits_unordered, key=lambda x: int(x.index))
    # target_logits = target_logits_unordered


    execution_time = time.time() - start_time
    logger.info(f"Target model execution time: {execution_time:.4f}s") # Added logging
    return target_logits_unordered, execution_time

def process_and_run_reward_model(prompts, completions, prm_client, prm_tokenizer, args, max_context_length=8192):
    """Process data and run the reward model, truncating inputs if necessary."""
    process_start_time = time.time()
    processed_data = [
        prepare_input(p, full_resp, tokenizer=prm_tokenizer, step_token="\n\n")
        for p, full_resp in zip(prompts, completions)
    ]
    input_ids_list, steps_list, reward_flags_list = zip(*processed_data)

    # Truncate inputs that exceed the maximum context length
    truncated_input_ids_list = []
    truncated_reward_flags_list = []
    for i, (input_ids, reward_flags) in enumerate(zip(input_ids_list, reward_flags_list)):
        if len(input_ids) > max_context_length:
            logger.warning(
                f"Input {i} length ({len(input_ids)}) exceeds max context length ({max_context_length}). Truncating from the beginning."
            )
            # Keep the end of the input, which includes the generated completion
            truncated_input_ids = input_ids[-max_context_length:]
            # Adjust reward flags accordingly
            truncated_reward_flags = reward_flags[-max_context_length:]
            # Ensure the first token isn't flagged if it wasn't originally
            # (though truncation from start makes this less likely to be an issue)
            if reward_flags[len(input_ids) - max_context_length] == 0:
                 truncated_reward_flags[0] = 0

        else:
            truncated_input_ids = input_ids
            truncated_reward_flags = reward_flags

        truncated_input_ids_list.append(truncated_input_ids)
        truncated_reward_flags_list.append(truncated_reward_flags)


    rewards = prm_client.embeddings.create(
        input=truncated_input_ids_list, # Use truncated inputs
        model=args.prm_path.split("/")[-1],
    )
    # Use original reward flags for deriving step rewards as truncation might affect indices
    # The derive_step_rewards_vllm function handles the mapping based on flags
    prm_scores = derive_step_rewards_vllm(rewards, truncated_reward_flags_list) # Use truncated flags
    prm_time = time.time() - process_start_time
    logger.info(f"PRM processing and execution time: {prm_time:.4f}s") # Added logging

    return prm_scores, prm_time

@dataclass
class Beam:
    prompt: str
    index: int
    current_text: str | None
    next_texts: list[str] | None
    lookahead_texts: list[str] | None
    stop_reasons: list[str | None] | None
    best_scores: list[float]  # the PRM scores
    all_scores: list[list[float]] | None # all PRM scores
    previous_text: str | None
    pruned: bool
    history: list[str]
    completed: bool = False
    completion_tokens: int = 0
    prob_scores: list = field(default_factory=list)
    reward_scores: list = field(default_factory=list)
    final_scores: list = field(default_factory=list)
    from_draft_model: list = field(default_factory=list)


@dataclass
class GenResult:
    index: int
    initial_prompt: str
    first_step_text: str
    first_step_stop_reason: str
    lookahead_text: str
    stop_reason: str | None
    cum_prob: float = 0.0
    step_rewards: list = field(default_factory=list)
    draft_cum_prob: float = 0
    num_generated_tokens: int = 0
    

def generate_k_steps_and_score(
    templated_convs,
    lookahead_steps: int,
    llm,
    sampling_params: SamplingParams,
    beam_width: int,
    llm_target = None,
    prm = None,
    target_tokenizer = None,
    prm_tokenizer = None,
    problem = "",
    config = None,
    use_target_as_draft = False
) -> list[Beam]:
    if llm_target is None:
        llm_target = llm


    gen_results = []
    for i, text in enumerate(templated_convs):
        for j in range(beam_width):
            gen_result = GenResult(
                index=i,
                initial_prompt=text,
                first_step_text="",
                lookahead_text="",
                stop_reason=None,
                first_step_stop_reason=None,
            )
            gen_results.append(gen_result)

    gen_sampling_params = copy.deepcopy(sampling_params)

    for i in range(lookahead_steps + 1):
        if i == 1:
            gen_sampling_params.temperature = 0.0  # greedy for the rest of the steps
        # get all generations that did not finish with eos
        current_gen = [
            gen_results[i]
            for i in range(len(gen_results)) # beam width 
            if gen_results[i].stop_reason != "EOS"
        ]
        gen_prompts = [
            gen_result.initial_prompt + gen_result.lookahead_text
            for gen_result in current_gen
        ]

        draft_start_time = time.time()

        if use_target_as_draft:
            draft_responses = llm_target.completions.create(
                model=config.target_model_path.split("/")[-1],
                prompt=gen_prompts,
                temperature=gen_sampling_params.temperature,
                top_p=gen_sampling_params.top_p,
                max_tokens=config.max_tokens,
                stop=gen_sampling_params.stop,
                n=1,
                stream=False,
                extra_body={
                    "include_stop_str_in_output": True
                },
                logprobs = 1
            ).choices
        else:
            draft_responses = llm.completions.create(
                model=config.draft_model_path.split("/")[-1],
                prompt=gen_prompts,
                temperature=gen_sampling_params.temperature,
                top_p=gen_sampling_params.top_p,
            max_tokens=config.max_tokens,
            stop=gen_sampling_params.stop,
            n=1,
            stream=False,
            extra_body={
                "include_stop_str_in_output": True
            },
            logprobs = 1
            ).choices

        draft_end_time = time.time()
        draft_time = draft_end_time - draft_start_time
        logger.debug(f"Draft time: {draft_time}")

        llm_outputs = sorted(draft_responses, key=lambda x: int(x.index))

        for gen_result, output in zip(current_gen, llm_outputs):
            gen_text = output.text
            if i == 0:
                gen_result.first_step_text = gen_text
                gen_result.first_step_stop_reason = output.stop_reason
                if gen_result.first_step_stop_reason is None:
                    gen_result.first_step_stop_reason = "EOS"

            gen_result.lookahead_text = gen_result.lookahead_text + gen_text
            gen_result.stop_reason = output.stop_reason
            if gen_result.stop_reason is None:
                gen_result.stop_reason = "EOS"

            # More efficient to get token count from the output's logprobs
            len_tokens_gen = len(output.logprobs.token_logprobs)
            draft_logprobs = output.logprobs.token_logprobs[:] # TODO: check for tokenized len mismatch
            gen_result.draft_cum_prob = torch.mean(torch.tensor(draft_logprobs))
            gen_result.num_generated_tokens = len_tokens_gen
    
    gen_prompts = [
        gen_result.initial_prompt + gen_result.lookahead_text
        for gen_result in gen_results
    ]
    gen_initial_prompts = [gen_result.initial_prompt for gen_result in gen_results]
    gen_lookahead_texts = [gen_result.lookahead_text for gen_result in gen_results]


    if config.speculative:
        # Define max context length, potentially fetch from config if available
        # Using 8192 as default based on the error message
        max_context_len = getattr(config, 'max_context_length', 8192)

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks to run in parallel
            # Pass target_tokenizer and max_context_len to run_target_model
            if use_target_as_draft:
                target_future = executor.submit(run_target_model, gen_prompts, llm, config, target_tokenizer, config.target_max_tokens, use_target_as_draft=use_target_as_draft)
            else:
                target_future = executor.submit(run_target_model, gen_prompts, llm_target, config, target_tokenizer, config.target_max_tokens, use_target_as_draft=use_target_as_draft)
            # Pass initial prompts and generated texts separately
            # Pass max_context_len to process_and_run_reward_model
            reward_future = executor.submit(process_and_run_reward_model, gen_initial_prompts, gen_lookahead_texts, prm, prm_tokenizer, config, max_context_len)


            # Get results (this will block until each result is ready)
            target_logits, target_execution_time = target_future.result()
            prm_scores, prm_time  = reward_future.result()

        for gen_result, output, step_rewards in zip(gen_results, target_logits, prm_scores, strict=True):
            len_tokens_gen = gen_result.num_generated_tokens
            target_logprobs = output.logprobs.token_logprobs[:] # TODO: check for tokenized len mismatch
            target_logprobs_pruned = [token_logprob for token_logprob in target_logprobs[-len_tokens_gen:] if token_logprob is not None and token_logprob != -9999.0]
            target_cum_prob = torch.mean(torch.tensor(target_logprobs_pruned))
            gen_result.cum_prob = target_cum_prob - gen_result.draft_cum_prob
            gen_result.step_rewards = step_rewards

    else:
        # Use the centralized function to process and run the reward model
        max_context_len = getattr(config, 'max_context_length', 8192) # Also get max length here
        prm_scores, prm_time = process_and_run_reward_model(
            gen_initial_prompts, gen_lookahead_texts, prm, prm_tokenizer, config, max_context_len # Pass max length
        )
        logger.info(f"PRM (non-speculative) processing and execution time: {prm_time:.4f}s")


        # Assign scores
        for gen_result, step_rewards in zip(gen_results, prm_scores):
            gen_result.cum_prob = 0.0 # No target model comparison
            gen_result.step_rewards = step_rewards
    
    outputs = []
    counter = 0
    for i, text in enumerate(templated_convs):
        next_texts = []
        stop_reasons = []
        prob_scores = []
        lookahead_texts = []
        step_rewards = []
        for j in range(beam_width):
            gen_result = gen_results[counter]
            next_texts.append(gen_result.first_step_text)
            stop_reasons.append(gen_result.first_step_stop_reason)
            prob_scores.append(gen_result.cum_prob.item() if hasattr(gen_result.cum_prob, 'item') else float(gen_result.cum_prob))
            lookahead_texts.append(gen_result.lookahead_text)
            step_rewards.append(gen_result.step_rewards)
            counter += 1
        beam_result = Beam(
            prompt=text,
            index=i,
            current_text="",
            next_texts=next_texts,
            lookahead_texts=lookahead_texts,
            stop_reasons=stop_reasons,
            best_scores=[0.0],
            all_scores=[],
            previous_text=None,
            pruned=False,
            history=[],
            prob_scores=prob_scores,
            reward_scores=step_rewards,
            final_scores=[]
        )
        outputs.append(beam_result)
    return outputs