#!/bin/bash
# Training script for BrowseComp-Plus with Qwen3-8B using FoldAgent

PROMPT_LENGTH=8192
RESPONSE_LENGTH=32768
MAX_LENGTH=40960
MODEL_PATH=Qwen/Qwen3-8B
TRAIN_DATA_PATH=data/bc_train.parquet
TEST_DATA_PATH=data/bc_test.parquet


python -m scripts.fold_agent_loop \
  algorithm.adv_estimator=foldgrpo \
  actor_rollout_ref.rollout.agent.default_agent_loop=fold_agent \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.model.path= ${MODEL_PATH} \
  actor_rollout_ref.rollout.prompt_length=${PROMPT_LENGTH} \
  actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  data.train_files=${TRAIN_DATA_PATH} \
  data.val_files=${TEST_DATA_PATH} \
  data.train_batch_size=32 \
  data.max_prompt_length=${PROMPT_LENGTH} \
  data.max_response_length=${RESPONSE_LENGTH} \
  data.return_raw_chat=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_LENGTH} \
  actor_rollout_ref.actor.ppo_infer_max_token_len_per_gpu=${MAX_LENGTH} \
  +actor_rollout_ref.rollout.plugin.workflow=search_branch \
  +actor_rollout_ref.rollout.plugin.max_turn=100 \
  +actor_rollout_ref.rollout.plugin.retry_cjk=10 \
  +actor_rollout_ref.rollout.plugin.turn_max_new_tokens=2048 \
  +actor_rollout_ref.rollout.plugin.max_session=10 \
  +actor_rollout_ref.rollout.plugin.val_max_session=10 \
  +actor_rollout_ref.rollout.plugin.session_timeout=3600 \
  +actor_rollout_ref.rollout.plugin.enable_summary=False \
  +actor_rollout_ref.rollout.plugin.branch_len=${RESPONSE_LENGTH} \
  +actor_rollout_ref.rollout.plugin.process_reward='[flat,scope]' \
  +actor_rollout_ref.rollout.plugin.max_traj=11 \
  +actor_rollout_ref.rollout.plugin.must_finish=False \
  +actor_rollout_ref.rollout.plugin.double_check=False \
  +actor_rollout_ref.rollout.plugin.must_search=True \
  +actor_rollout_ref.rollout.plugin.val_max_turn=100 \
  +actor_rollout_ref.rollout.plugin.val_response_length=${RESPONSE_LENGTH} \
  trainer.val_before_train=False \
  trainer.val_only=False \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.total_training_steps=100 \
  trainer.test_freq=10 \
  trainer.save_freq=10 \
  trainer.project_name=context_folding \
  trainer.experiment_name=test_run
