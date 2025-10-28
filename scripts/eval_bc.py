#!/usr/bin/env python3
import asyncio
import argparse
import json
import numpy as np
import pandas as pd
import sys
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings('ignore', message='.*fast tokenizer.*')

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.fold import process_item
from agents.utils import CallAPI, TaskContext
from verl import DataProto
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate agents on BrowseComp-Plus benchmark')
    parser.add_argument('--data_path', default='data/bc_test.parquet',
                        help='Path to test data parquet file (default: data/bc_test.parquet)')
    parser.add_argument('--output_dir', default='results',
                        help='Directory to save evaluation results (default: results)')
    parser.add_argument('--prompt_length', type=int, default=16384,
                        help='Maximum prompt length in tokens (default: 16384)')
    parser.add_argument('--response_length', type=int, default=32768,
                        help='Maximum response length in tokens (default: 32768)')
    parser.add_argument('--workflow', default='search_branch',
                        help='Agent workflow: "search" for ReAct, "search_branch" for Context-Folding (default: search_branch)')
    parser.add_argument('--max_turn', type=int, default=200,
                        help='Maximum turns during training (default: 200)')
    parser.add_argument('--val_max_turn', type=int, default=200,
                        help='Maximum turns during validation/evaluation (default: 200)')
    parser.add_argument('--max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during training (default: 10)')
    parser.add_argument('--val_max_session', type=int, default=10,
                        help='Maximum branch sessions for Context-Folding during validation (default: 10)')
    parser.add_argument('--model_name', default='gpt-5-nano',
                        help='Model name for API (e.g., gpt-5-nano, gpt-4o, or vLLM model path) (default: gpt-5-nano)')
    parser.add_argument('--num_workers', type=int, default=150,
                        help='Number of parallel evaluation workers (default: 150)')
    parser.add_argument('--local_search_url', default='http://localhost:8000',
                        help='URL of the local search server (default: http://localhost:8000)')
    parser.add_argument('--enable_summary', action='store_true',
                        help='Enable summary mode (use with workflow=search for Summary agent)')
    return parser.parse_args()


async def eval_one(row, config, tokenizer, model_name):
    context = TaskContext(config=config, global_step=0, server_host=model_name,
                          server_port=0, is_train=False, tokenizer=tokenizer)

    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id', 'unknown')], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {'generation_kwargs': {}, 'max_turn': config.actor_rollout_ref.rollout.plugin.val_max_turn}

    output = await process_item(item, context, CallAPI)

    result = {
        'instance_id': row['extra_info'].get('instance_id', 'unknown'),
        'data_source': row.get('data_source', 'unknown'),
        'score': output.non_tensor_batch.get('extra_data', [{}])[0].get('stats', {}).get('score', 0) if output else 0,
        'status': 'success' if output else 'failed'
    }
    return result


async def worker(worker_id, rows, args, pbar, shared_scores):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

    config = OmegaConf.create({
        'actor_rollout_ref': {'rollout': {
            'prompt_length': args.prompt_length,
            'response_length': args.response_length,
            'plugin': {
                'workflow': args.workflow,
                'max_turn': args.max_turn,
                'val_max_turn': args.val_max_turn,
                'max_session': args.max_session,
                'val_max_session': args.val_max_session,
                'session_timeout': 5400,
                'process_reward': None,
                'max_traj': None,
                'must_finish': False,
                'double_check': False,
                'must_search': False,
                'enable_summary': args.enable_summary
            }
        }}
    })

    results = []
    for row in rows:
        result = await eval_one(row, config, tokenizer, args.model_name)
        results.append(result)
        shared_scores.append(result['score'])
        avg_score = np.mean(shared_scores)
        pbar.set_postfix({'avg_score': f"{avg_score:.3f}", 'id': result['instance_id']})
        pbar.update(1)

    return results


def main():
    args = parse_args()
    os.environ["LOCAL_SEARCH_URL"] = args.local_search_url

    # Load data
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} items")

    # Split for workers
    chunk_size = len(df) // args.num_workers
    chunks = [df.iloc[i*chunk_size:(i+1)*chunk_size if i < args.num_workers-1 else len(df)]
              for i in range(args.num_workers)]

    # Run workers with progress bar
    async def run_all():
        shared_scores = []
        with tqdm(total=len(df), desc="Evaluating", unit="item") as pbar:
            tasks = [worker(i, [chunks[i].iloc[j] for j in range(len(chunks[i]))], args, pbar, shared_scores)
                     for i in range(args.num_workers)]
            return await asyncio.gather(*tasks)

    all_results = asyncio.run(run_all())
    results = [r for worker_results in all_results for r in worker_results]

    # Summary overall
    avg_score = np.mean([r['score'] for r in results])
    print(f"\n{'='*60}")
    print(f"Overall - Avg Score: {avg_score:.4f}, Success: {sum(r['status']=='success' for r in results)}/{len(results)}")

    # Summary by data_source
    from collections import defaultdict
    by_source = defaultdict(list)
    for r in results:
        by_source[r['data_source']].append(r['score'])

    print(f"\nBy Data Source:")
    for source in sorted(by_source.keys()):
        scores = by_source[source]
        print(f"  {source}: {np.mean(scores):.4f} ({len(scores)} items)")

    # Save
    Path(args.output_dir).mkdir(exist_ok=True)
    output_file = Path(args.output_dir) / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_by_source = {src: {'avg_score': float(np.mean(scores)), 'count': len(scores)}
                         for src, scores in by_source.items()}
    json.dump({'avg_score': avg_score, 'by_source': summary_by_source, 'results': results},
              open(output_file, 'w'), indent=2)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()