# Scaling Long-Horizon LLM Agent via Context-Folding

Paper: https://arxiv.org/pdf/2510.11967

<img width="4239" height="4110" alt="cover" src="https://github.com/user-attachments/assets/9c9c0b67-ccd8-4a4b-859b-22f613f41954" />

## Training
Coming soon

## Evaluation
**Start Search Server:**
```bash
cd envs && python search_server.py \
  --model Qwen/Qwen3-Embedding-8B \
  --corpus Tevatron/browsecomp-plus-corpus \
  --corpus-embedding-dataset miaolu3/browsecomp-plus \
  --host 0.0.0.0 \
  --port 8000
```

**Using OpenAI API:**
```bash
export OPENAI_API_KEY='your-key'

python scripts/eval_bc.py \
  --model_name gpt-5-nano \
  --num_workers 4 \
  --prompt_length 16384 \
  --response_length 128000 \
  --workflow search \
  --max_turn 500 \
  --val_max_turn 500 \
  --max_session 10 \
  --val_max_session 10 \
  --local_search_url http://localhost:8000 \
  --output_dir results
```

**Using Open-Source Models (vLLM):**
```bash
# Start vLLM server
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8001 --max-model-len 131072

# Run evaluation
export OPENAI_API_KEY='dummy'
export OPENAI_BASE_URL='http://localhost:8001/v1'

python scripts/eval_bc.py \
  --model_name Qwen/Qwen2.5-32B-Instruct \
  --num_workers 10 \
  --prompt_length 16384 \
  --response_length 128000 \
  --workflow search \
  --max_turn 500 \
  --val_max_turn 500 \
  --max_session 10 \
  --val_max_session 10 \
  --local_search_url http://localhost:8000 \
  --output_dir results
```

**Output:**
```
Evaluating: 100%|████████| 100/100 [15:23<00:00, score=0.85]
Overall - Avg Score: 0.7542, Success: 98/100
By Data Source:
  bc_test_easy: 0.89 (45 items)
  bc_test_medium: 0.71 (25 items)
  bc_test_hard: 0.62 (30 items)
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
