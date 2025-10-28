# Scaling Long-Horizon LLM Agent via Context-Folding

Paper: https://arxiv.org/pdf/2510.11967

<img width="4239" height="4110" alt="cover" src="https://github.com/user-attachments/assets/9c9c0b67-ccd8-4a4b-859b-22f613f41954" />

## Training
Coming soon

## Evaluation

### Start Search Server
```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 \
  --port 8000
```

### Evaluate on BrowseComp

- Download and decompress: https://drive.google.com/file/d/1aX5xXAN5R-gLKd8A0AY-troxXJRawyAM/view?usp=sharing

- **Fold Agent:** `workflow=search_branch`
```bash
export OPENAI_API_KEY='your-key'

python scripts/eval_bc.py \
  --data_path data/bc_test.parquet \
  --model_name gpt-5-nano \
  --num_workers 150 \
  --workflow search_branch \
  --prompt_length 16384 \
  --response_length 32768 \
  --max_turn 200 \
  --val_max_turn 200 \
  --max_session 10 \
  --val_max_session 10 \
  --local_search_url http://localhost:8000 \
  --output_dir results
```
Output:
```
Evaluating: 100%|█████████████| 150/150 [32:52<00:00, 13.15s/item, avg_score=0.407, id=122]

============================================================
Overall - Avg Score: 0.4067, Success: 150/150

By Data Source:
  bc_test_easy: 0.8200 (50 items)
  bc_test_hard: 0.0400 (50 items)
  bc_test_meduim: 0.3600 (50 items)
```

- **ReAct Agent:**  `workflow=search`
```bash
python scripts/eval_bc.py --workflow search [...]
```

- **Summary Agent:** `workflow=search, enable_summary`
```bash
python scripts/eval_bc.py --workflow search --enable_summary [...]
```

### Using vLLM
```bash
# Start vLLM server
vllm serve ByteDance-Seed/Seed-OSS-36B-Instruct --port 8001 --max-model-len 131072

# Run evaluation
export OPENAI_API_KEY='dummy'
export OPENAI_BASE_URL='http://localhost:8001/v1'

python scripts/eval_bc.py \
  --model_name ByteDance-Seed/Seed-OSS-36B-Instruct \
  --workflow search_branch \
  --num_workers 32 \
  --prompt_length 16384 \
  --response_length 32768 \
  --max_turn 100 \
  --val_max_turn 100 \
  --max_session 10 \
  --val_max_session 10 \
  --output_dir results
```

### Evaluation on SWE-Bench Verified
Coming soon

## Cite

```
@article{sun2025scaling,
  title   = {Scaling Long-Horizon LLM Agent via Context-Folding},
  author  = {Sun, Weiwei and Lu, Miao and Ling, Zhan and Liu, Kang and Yao, Xuesong and Yang, Yiming and Chen, Jiecao},
  journal = {arXiv preprint arXiv:2510.11967},
  year    = {2025},
}
```
