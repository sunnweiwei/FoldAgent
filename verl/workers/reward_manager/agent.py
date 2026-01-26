# Copyright 2025 VERL Contributors
# Licensed under the Apache License, Version 2.0
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("agent_loop")
class AgentLoopRewardManager(AbstractRewardManager):
    """Reward manager for agent-loop outputs.

    - If `rm_scores` are present in `DataProto.batch`, use them directly.
    - Otherwise, decode prompts/responses and call `compute_score` to produce scalar rewards.
    - Computes batch-level metrics with de-duplication by `gen_uid` (multiple trajectories per rollout).
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If rm_scores exist (from agent_loop postprocess), prefer them
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_tensor = data.batch["rm_scores"].to(torch.float32)
                # Per-sample scalar (last valid token) for logging
                per_sample_scores = []
                prompts = data.batch["prompts"]
                attention_mask = data.batch["attention_mask"]
                prompt_len = prompts.size(1)
                for i in range(len(data)):
                    valid_resp_len = int(attention_mask[i, prompt_len:].sum().item())
                    per_sample_scores.append(float(reward_tensor[i, valid_resp_len - 1].item()))
                # Batch metrics with de-dup by gen_uid
                reward_extra_info = self._compute_batch_metrics(data, per_sample_scores)
                # Also attach per-sample scores
                reward_extra_info["reward_score"] = np.array(per_sample_scores)
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"].to(torch.float32)

        # Otherwise compute rewards via compute_score
        prompts = data.batch["prompts"]
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        prompt_len = prompts.size(1)
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)

        per_sample_scores: list[float] = []
        for i in range(len(data)):
            valid_prompt_len = int(attention_mask[i, :prompt_len].sum().item())
            valid_resp_len = int(attention_mask[i, prompt_len:].sum().item())

            valid_prompt_ids = prompts[i, -valid_prompt_len:]
            valid_resp_ids = responses[i, :valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            # Required fields
            ground_truth = data[i].non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
            data_source = data[i].non_tensor_batch.get(self.reward_fn_key)
            extra_info = dict(data[i].non_tensor_batch.get("extra_info", {}))
            # Add optional context
            extra_info["num_turns"] = data[i].non_tensor_batch.get("__num_turns__", None)

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if isinstance(score, dict):
                reward = float(score.get("score", 0.0))
            else:
                reward = float(score)

            reward_tensor[i, valid_resp_len - 1] = reward
            per_sample_scores.append(reward)

            # Optional debug printing
            if self.num_examine and i < self.num_examine:
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score)

        if return_dict:
            reward_extra_info = self._compute_batch_metrics(data, per_sample_scores)
            reward_extra_info["reward_score"] = np.array(per_sample_scores)
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor

    def _compute_batch_metrics(self, data: DataProto, per_sample_scores: list[float]) -> dict[str, Any]:
        """Compute batch-level metrics with de-duplication by gen_uid.

        - Average score over unique gen_uid, taking max per gen_uid across trajectories
        - Std/min/max
        - Number of unique gen_uids
        - Average trajectories per gen_uid
        - Overlong rate across unique gen_uids (any trajectory masked)
        - Average num turns (mean over per-gen_uid max of num_turns)
        """
        gen_uid_list = data.non_tensor_batch.get("gen_uid")
        if gen_uid_list is None:
            gen_uid_list = np.arange(len(data))
        gen_uid_list = gen_uid_list.tolist() if hasattr(gen_uid_list, "tolist") else list(gen_uid_list)

        # Exclude dummy trajectories by exact gen_uid match (single UUID)
        gen_uid_dummy = data.meta_info.get("gen_uid_dummy", None)
        if gen_uid_dummy is not None and len(gen_uid_list) > 0:
            keep_indices = [
                i for i, uid in enumerate(gen_uid_list)
                # Support uuid objects or stringified uuids
                if uid != gen_uid_dummy and str(uid) != str(gen_uid_dummy)
            ]
            gen_uid_list = [gen_uid_list[i] for i in keep_indices]
            per_sample_scores = [per_sample_scores[i] for i in keep_indices]
            print("exclude dummy gen_uid", gen_uid_dummy, "num of kept trajectories", len(keep_indices))

        gen_uid_to_scores: dict[Any, list[float]] = defaultdict(list)
        for gen_uid, score in zip(gen_uid_list, per_sample_scores):
            gen_uid_to_scores[gen_uid].append(score)

        per_gen_uid_max_scores = [max(scores) for scores in gen_uid_to_scores.values()] if gen_uid_to_scores else []

        def _safe_mean(xs: list[float]) -> float:
            return float(sum(xs) / max(1, len(xs)))
        def _safe_std(xs: list[float]) -> float:
            if len(xs) <= 1:
                return 0.0
            m = _safe_mean(xs)
            return float((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5)

        avg_score = _safe_mean(per_gen_uid_max_scores)
        std_score = _safe_std(per_gen_uid_max_scores)
        min_score = float(min(per_gen_uid_max_scores)) if per_gen_uid_max_scores else 0.0
        max_score = float(max(per_gen_uid_max_scores)) if per_gen_uid_max_scores else 0.0
        num_unique_gen_uids = len(gen_uid_to_scores)
        avg_trajs_per_gen_uid = _safe_mean([len(v) for v in gen_uid_to_scores.values()]) if gen_uid_to_scores else 0.0

        overlong = data.non_tensor_batch.get("mask_rollout", None)
        overlong_rate = None
        if overlong is not None:
            overlong_list = overlong.tolist() if hasattr(overlong, "tolist") else list(overlong)
            gen_uid_to_overlong_any: dict[Any, bool] = defaultdict(lambda: False)
            for gen_uid, flag in zip(gen_uid_list, overlong_list):
                gen_uid_to_overlong_any[gen_uid] = bool(gen_uid_to_overlong_any[gen_uid] or flag)
            overlong_rate = _safe_mean([float(v) for v in gen_uid_to_overlong_any.values()])

        # Num turns: sum across trajectories per gen_uid
        num_turns_list = data.non_tensor_batch.get("__num_turns__", None)
        avg_num_turns = None
        if num_turns_list is not None:
            num_turns_list = num_turns_list.tolist() if hasattr(num_turns_list, "tolist") else list(num_turns_list)
            gen_uid_to_num_turns: dict[Any, list[float]] = defaultdict(list)
            for gen_uid, nt in zip(gen_uid_list, num_turns_list):
                gen_uid_to_num_turns[gen_uid].append(float(nt))
            per_gen_uid_sum_num_turns = [sum(v) for v in gen_uid_to_num_turns.values()] if gen_uid_to_num_turns else []
            avg_num_turns = _safe_mean(per_gen_uid_sum_num_turns)

        bsz = len(data)
        repeat = lambda x: np.array([x] * bsz) if x is not None else np.array([None] * bsz)
        reward_extra_info = {
            "avg_score": repeat(avg_score),
            "std_score": repeat(std_score),
            "min_score": repeat(min_score),
            "max_score": repeat(max_score),
            "num_unique_gen_uids": repeat(float(num_unique_gen_uids)),
            "avg_trajs_per_gen_uid": repeat(avg_trajs_per_gen_uid),
            "overlong_rate": repeat(overlong_rate),
            "avg_num_turns": repeat(avg_num_turns),
        }
        return reward_extra_info