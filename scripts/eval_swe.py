#!/usr/bin/env python3
"""Evaluate agents on SWE-Bench Verified using per-instance Modal sandboxes."""

import argparse
import asyncio
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", message=".*fast tokenizer.*")

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from agents.fold_agent import process_item
from agents.utils import CallAPI, TaskContext
from verl import DataProto


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate agents on SWE-Bench via Modal sandboxes")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified",
                   help="HuggingFace dataset for SWE-Bench")
    p.add_argument("--split", default="test")
    p.add_argument("--instance_ids", default=None,
                   help="Comma-separated subset of instance_ids (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of instances")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--prompt_length", type=int, default=32768)
    p.add_argument("--response_length", type=int, default=65536)
    p.add_argument("--workflow", default="code_branch",
                   help="`code` for ReAct, `code_branch` for Context-Folding")
    p.add_argument("--max_turn", type=int, default=200)
    p.add_argument("--val_max_turn", type=int, default=200)
    p.add_argument("--max_session", type=int, default=10)
    p.add_argument("--val_max_session", type=int, default=10)
    p.add_argument("--model_name", default="gpt-5-nano")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--session_timeout", type=int, default=5400)
    p.add_argument("--image_namespace", default="swebench")
    p.add_argument("--image_tag", default="latest")
    p.add_argument("--image_arch", default="x86_64")
    p.add_argument("--app_name", default="foldagent-swe")
    return p.parse_args()


def _load_instances(dataset: str, split: str, instance_ids: str | None, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    if instance_ids:
        wanted = {s.strip() for s in instance_ids.split(",") if s.strip()}
        rows = [dict(r) for r in ds if r["instance_id"] in wanted]
    else:
        rows = [dict(r) for r in ds]
    if limit:
        rows = rows[:limit]
    return rows


def _ability_from_instance(inst: dict) -> str:
    keep_keys = (
        "instance_id", "repo", "version", "base_commit", "problem_statement",
        "patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "hints_text",
        "environment_setup_commit",
    )
    slim = {k: inst[k] for k in keep_keys if k in inst}
    return "SWEModalEnv@" + json.dumps(slim)


async def eval_one(inst, config, tokenizer, model_name):
    llm_client = CallAPI(url=model_name, tokenizer=tokenizer,
                         config=config.actor_rollout_ref.rollout)
    context = TaskContext(config=config, global_step=0, llm_client=llm_client,
                          is_train=False, tokenizer=tokenizer)

    item = DataProto()
    item.non_tensor_batch = {
        "ability": np.array([_ability_from_instance(inst)], dtype=object),
        "extra_info": np.array([{
            "instance_id": inst["instance_id"],
            "dataset_id": "swe-bench-verified",
            "workflow": None,
        }], dtype=object),
        "uid": np.array([inst["instance_id"]], dtype=object),
        "reward_model": np.array([{}], dtype=object),
    }
    item.meta_info = {
        "generation_kwargs": {},
        "max_turn": config.actor_rollout_ref.rollout.plugin.val_max_turn,
    }

    output = await process_item(item, context)
    score = 0
    report: dict = {}
    env_stats: dict = {}
    num_turns = 0
    is_finish = False
    if output:
        first = output[0] if isinstance(output, list) else output
        score = getattr(first, "reward_score", 0) or 0
        num_turns = getattr(first, "num_turns", 0)
        extra = getattr(first, "extra_fields", {}) or {}
        env_stats = extra.get("env_stats", {}) or {}
        report = env_stats.get("eval_report", {}) or {}
        is_finish = bool(extra.get("is_finish", False))
    return {
        "instance_id": inst["instance_id"],
        "repo": inst.get("repo", ""),
        "score": float(score),
        "resolved": bool(report.get("resolved")),
        "is_finish": is_finish,
        "num_turns": num_turns,
        "env_stats": {k: v for k, v in env_stats.items() if k != "eval_report"},
        "report": report,
    }


async def worker(rows, args, pbar, shared):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                              trust_remote_code=True)
    config = OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "prompt_length": args.prompt_length,
            "response_length": args.response_length,
            "plugin": {
                "workflow": args.workflow,
                "max_turn": args.max_turn,
                "val_max_turn": args.val_max_turn,
                "max_session": args.max_session,
                "val_max_session": args.val_max_session,
                "session_timeout": args.session_timeout,
                "process_reward": None,
                "max_traj": None,
                "must_finish": False,
                "double_check": False,
                "must_search": False,
                "enable_summary": False,
            },
        }},
        "trainer": {"agent_eval_timeout": 1500},
    })

    os.environ["SWE_APP_NAME"] = args.app_name
    os.environ["SWE_IMAGE_NAMESPACE"] = args.image_namespace
    os.environ["SWE_IMAGE_TAG"] = args.image_tag
    os.environ["SWE_IMAGE_ARCH"] = args.image_arch

    results = []
    for row in rows:
        try:
            r = await eval_one(row, config, tokenizer, args.model_name)
        except Exception as e:
            r = {"instance_id": row["instance_id"], "score": 0.0,
                 "resolved": False, "error": str(e)}
        results.append(r)
        shared.append(r["score"])
        pbar.set_postfix({
            "resolved": f"{sum(1 for x in shared if x > 0)}/{len(shared)}",
            "id": r["instance_id"],
        })
        pbar.update(1)
    return results


def main():
    args = parse_args()
    instances = _load_instances(args.dataset, args.split, args.instance_ids, args.limit)
    print(f"Loaded {len(instances)} instances")

    n = max(1, min(args.num_workers, len(instances)))
    chunks = [instances[i::n] for i in range(n)]

    async def run_all():
        shared = []
        with tqdm(total=len(instances), desc="Eval", unit="inst") as pbar:
            tasks = [worker(chunks[i], args, pbar, shared) for i in range(n)]
            return await asyncio.gather(*tasks)

    grouped = asyncio.run(run_all())
    results = [r for g in grouped for r in g]

    resolved = sum(1 for r in results if r.get("resolved"))
    print(f"\nResolved: {resolved}/{len(results)} = {resolved/max(1,len(results)):.4f}")

    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    out = Path(args.output_dir) / f"swe_results_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(out, "w") as f:
        json.dump({
            "dataset": args.dataset,
            "split": args.split,
            "model": args.model_name,
            "workflow": args.workflow,
            "resolved": resolved,
            "total": len(results),
            "resolve_rate": resolved / max(1, len(results)),
            "results": results,
        }, f, indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()