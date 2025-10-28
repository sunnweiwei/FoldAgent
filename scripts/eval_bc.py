import asyncio
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from transformers import AutoTokenizer
from agents.fold import process_item
from agents.utils import CallAPI, TaskContext
from verl import DataProto
import os

os.environ["LOCAL_SEARCH_URL"] = "http://localhost:8000"
df = pd.read_parquet("data/bc_test.parquet")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# Create a mock config for evaluation
config_dict = {
    'actor_rollout_ref': {
        'rollout': {
            'prompt_length': 16384,
            'response_length': 128000,
            'plugin': {
                'workflow': 'search',
                'max_turn': 500,
                'val_max_turn': 500,
                'max_session': 10,
                'val_max_session': 10,
                'session_timeout': 5400,
                'process_reward': None,
                'max_traj': None,
                'must_finish': False,
                'double_check': False,
                'must_search': True,
            }
        }
    }
}
config = OmegaConf.create(config_dict)

# Create context
context = TaskContext(
    config=config,
    global_step=0,
    server_host='gpt-5-mini',
    server_port=0,
    is_train=False,
    tokenizer=tokenizer
)

async def evaluate_item(row):
    item = DataProto()
    item.non_tensor_batch = {
        'ability': np.array([row['ability']], dtype=object),
        'extra_info': np.array([row['extra_info']], dtype=object),
        'uid': np.array([row['extra_info'].get('instance_id', 'unknown')], dtype=object),
        'reward_model': np.array([row['reward_model']], dtype=object),
    }
    item.meta_info = {
        'generation_kwargs': {},
        'max_turn': 64,
    }
    output = await process_item(item, context, CallAPI)
    return output

# Main evaluation loop
for idx, row in df.iterrows():
    print(row)
    if row['data_source'] != 'bc_test_easy':
        continue
    input("Press Enter to continue...")

    output = asyncio.run(evaluate_item(row))

    print(f"\nCompleted item {idx + 1}")
    if output is not None:
        print(f"Output meta_info: {output.meta_info.get('xperf_metrics', {})}")


