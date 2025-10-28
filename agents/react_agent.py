import os
import re
import time
import copy
import asyncio
from functools import partial
import random

import numpy as np
import torch

from verl import DataProto
from .utils import CallLLM, Agent, select_env, truncate_text, is_weird, TaskContext, CallAPI
from .prompts import create_chat


async def run_action(env, response):
    try:
        try:
            act = time.time()
            env_return = await asyncio.wait_for(env.run_action(response), timeout=120.0)
            if time.time() - act > 10:
                print('Action Cost', time.time() - act)
        except asyncio.TimeoutError:
            print('[ACTION] Action timed out after 120 seconds')
            env_return = {'observation': 'Action timed out after 120 seconds'}
        if 'action' in env_return:
            action, arguments = env_return['action'], env_return.get('arguments', {})
            if action == 'finish':
                return None
        observation = env_return.pop('observation', 'Empty')
    except Exception as e:
        observation = f"Error: {e}"
    return observation


async def process_item(
        item: DataProto,
        context: TaskContext,
        LLMClass=CallLLM,
) -> DataProto:
    os.environ["no_proxy"] = ""
    tokenizer = context.tokenizer
    config = context.config.actor_rollout_ref.rollout
    is_train = context.is_train

    if not is_train:
        if getattr(config.plugin, "val_response_length", None):
            config.response_length = getattr(config.plugin, "val_response_length", None)
    ability = item.non_tensor_batch['ability'][0]
    # Select env
    EnvClass = select_env(ability, config, )
    print(is_train, EnvClass)
    env = EnvClass(config, tokenizer, ability)

    try:
        await env.init_env(item)
    except Exception as e:
        print(f"[Error] during environment init: {str(e)}")

    user_prompt, agent_config = await env.get_data(item, context)
    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                       "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = agent_config.get("max_turn", 64)
    host = context.server_host
    port = context.server_port
    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))
    prompt_turn = len(user_prompt)

    agent = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    iteration = 0
    while iteration < max_turn:
        iteration += 1
        response = await agent.step()
        if response is None:
            break
        observation = await run_action(env, response)
        if observation is None:
            break
        agent.append({'role': 'user', 'content': observation})

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent.messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    out = await agent.dataproto()
    messages = agent.messages()
    out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                         tag='main', metrics=agent.get_metrics())
    res = DataProto.concat([out])
    return res


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
