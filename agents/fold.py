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
from .prompts import BRANCH_MESSAGE_SEARCH, BRANCH_MESSAGE
from .verifier import judge_scope


def print_chat(chat):
    chat_str = ""
    for turn in chat:
        if is_weird(str(turn)):
            chat_str += '# ' + turn['role'] + ' **CJK**\n\n' + turn['content'] + "\n\n---\n\n"
        else:
            chat_str += '# ' + turn['role'] + '\n\n' + turn['content'] + "\n\n---\n\n"
    return chat_str

def extract_fn_call(text):
    if text is None:
        return None
    func_matches = re.findall(r'<function=([^>]+)>', text)
    if not func_matches:
        return None
    last_function = func_matches[-1]
    last_func_pos = text.rfind(f'<function={last_function}>')
    text_after_last_func = text[last_func_pos:]
    params = dict(re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', text_after_last_func, re.DOTALL))
    return {'function': last_function, 'arguments': params}


def clean_response(response):
    if '<function=return>' in response:
        response = response.split('<function=return>')[-1]
    else:
        response = re.split(r'<\[[^\]]+\]>', response)[-1]
    return response



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

    if 'search' in workflow:
        branch_prompt = BRANCH_MESSAGE_SEARCH
    else:
        branch_prompt = BRANCH_MESSAGE

    max_turn = agent_config.get("max_turn", 64)
    max_session = getattr(config.plugin, "max_session", 5)
    if not is_train:
        max_session = getattr(config.plugin, "val_max_session", max_session)
    session_timeout = getattr(config.plugin, "session_timeout", 90 * 60)
    process_reward = getattr(config.plugin, "process_reward", None)
    if process_reward is not None and isinstance(process_reward, str) and process_reward.lower() == "none":
        process_reward = None
    max_traj = getattr(config.plugin, "max_traj", None)

    host = context.server_host
    port = context.server_port

    llm_client = LLMClass(host, port, tokenizer, config, meta_info=agent_config.get("meta_info", {}))

    prompt_turn = len(user_prompt)
    agent = dict()
    agent['main'] = Agent(llm_client, user_prompt, tokenizer, config, prompt_turn=prompt_turn)
    branches = []
    branch_tasks = {}
    branch_return = {}
    init_len = len(agent['main'].context())
    current = 'main'
    session_start_time = time.time()
    iteration = 0
    mask_rollout = True  # If True then no grad update on this traj
    session_message = []
    while iteration < max_turn:
        if time.time() - session_start_time > session_timeout:
            print('[SESSION] Session Timeout')
            break

        iteration += 1

        response = await agent['main'].step()
        # print(response)

        if response is None:
            break

        session_message.append({'role': 'assistant', 'content': response})
        fn_call = extract_fn_call(response)
        if fn_call is not None and fn_call['function'] == 'branch':
            if len(branches) + 1 > max_session:
                observation = f"You've already reached the limit of {len(branches)} branch calls. Continue working independently."
            else:
                description = fn_call['arguments'].get('description', 'Agent')
                message_to_branch = fn_call['arguments'].get('prompt', 'Empty prompt')
                print('[BRANCH]', description, len(agent['main'].context()))
                agent_name = f"#{len(branches)}-" + description.replace(' ', '_')
                branches.append(agent_name)
                branch_tasks[agent_name] = message_to_branch
                history = agent['main'].messages()
                agent[agent_name] = Agent(llm_client, history, tokenizer, config, prompt_turn=prompt_turn)
                branch_prompt_formatted = branch_prompt.format(message=message_to_branch)
                agent[agent_name].append({'role': 'user', 'content': branch_prompt_formatted})
                agent_return = await agent[agent_name].react(
                    partial(run_action, env),
                    max_turn=max_turn,
                    max_tokens=getattr(config.plugin, "branch_len", None),
                    session_timeout=session_timeout - time.time() + session_start_time,
                    should_continue=lambda resp: '<function=return>' not in resp,
                    safe_finish=lambda
                        x: "You are in branch mode and cannot branch task or finish the task. Use the `return` tool to go back to the main agent." if '<function=finish>' in x or '<function=branch>' in x else None,
                    summary_prompt="The context limit has been exceeded for the branch. Please finish the sub task directly and clearly state the progress made and the pending jobs of the sub task. Only summarize the sub task progress, using the return tool.",
                    observation_prompt=f"* You are now in branch mode: {description}. Conduct the sub task based on instruction, and when you complete the assigned sub task, use return tool to return, do not perform action beyond the assigned sub task.",
                )
                iteration += agent_return['iteration']
                last_response = agent_return['last_response']
                session_message.extend(agent[agent_name].messages()[len(history):])
                fn_call = extract_fn_call(last_response)
                branch_message = None
                if fn_call is not None and fn_call['function'] == 'return':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                elif fn_call is not None and fn_call['function'] == 'finish':
                    if 'message' in fn_call['arguments']:
                        branch_message = fn_call['arguments'].get('message', 'Empty message')
                        branch_message = f'Branch has finished its task, the returned message is:\n\n{branch_message}'
                if branch_message is None:
                    branch_message = f'Branch has finished its task. The last message was:\n\n{clean_response(last_response)}'
                observation = branch_message
                branch_return[agent_name] = observation
                # print(observation)
        else:
            observation = await run_action(env, response)
            if observation is None:
                mask_rollout = False
                break

        if agent['main'].chat[-1]['role'] == 'user':
            print('[ROLE ERROR]')
            print(agent['main'].chat[-1])
            agent['main'].append({'role': 'assistant', 'content': str(response)})

        if process_reward:
            observation = truncate_text(observation, max_lines=100, merge_repeat=True, merge_num=4)
        # print(observation)
        agent['main'].append({'role': 'user', 'content': observation})
        session_message.append({'role': 'user', 'content': observation})

    env.stats['session_time'] = time.time() - session_start_time

    print('[TASK] Task Finish, Start Reward')
    try:
        score_msg, reward, reward_dict = await asyncio.wait_for(
            env.get_reward(item, agent['main'].messages(), context), timeout=60 * 10)
        score = (score_msg, reward)
        print(score)
    except Exception as e:
        print(f"[Error] Getting reward: {e}")
        score, reward_dict = ("", 0), {"ans_reward": 0.0, "format_reward": 0.0, "ref_reward": 0.0}

    outs = []
    env.stats['get_final_score'] = score[1]
    env.stats['traj_num'] = len(agent)
    env.stats['main_len'] = min(len(agent['main'].context()) - init_len, config.response_length)
    env.stats['total_token'] = len(tokenizer.encode(print_chat(user_prompt + session_message)))
    env.stats['main_turn'] = len(agent['main'].messages())
    env.stats['is_branch'] = int(len(agent) > 1)
    env.stats['branch_success'] = int(int(len(agent) > 1) * score[1])
    env.stats['use_all_branch'] = int(len(branches) + 1 > max_session)

    if getattr(env, 'is_finish', False) or getattr(env, 'finish', False):
        mask_rollout = False
    if score[1] > 0:
        mask_rollout = False

    is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
    if getattr(config.plugin, "must_finish", None):
        if not is_finish:
            score = ('', 0)

    if process_reward and is_train:
        mask_rollout = False
        env.stats['concise_main'] = int(len(agent['main'].context()) - init_len <= config.response_length * 0.5)
        if 'cjk' in process_reward:
            env.stats['is_cjk'] = 0
            for name in agent:
                for i, turn in enumerate(agent[name].chat):
                    if is_weird(str(turn)):
                        print('[CJK ERROR]')
                        print(str(turn))
                        env.stats['is_cjk'] = 1
                        agent[name].set_process_reward(i, -1)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0)
        if score[1] > 0:
            # Check main
            if len(agent['main'].context()) - init_len > config.response_length * 0.5:
                bad_turn = [i for i, turn in enumerate(agent['main'].messages()) if
                            '<function=branch>' not in str(turn) and '<function=finish>' not in str(turn)]
                agent['main'].set_process_reward(bad_turn, -1)

            if len(agent) == 1:
                agent['main'].set_process_reward('all', -1)
                if 'flat' in process_reward:
                    agent['main'].set_cache('reward', 1 - 1)

            # Scope check
            if 'scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:  # scope check, skip summary turn
                        print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 1 + 0.2)
            # Tool call error
            for name in branches:
                for i, turn in enumerate(agent[name].chat):
                    ERR_MARKERS = (
                        'Failed to validate tool call',
                        'Failed to parse tool call',
                        'You are in branch mode and cannot branch task or finish the task.',
                        'No function call was detected in the model response',
                        '[Error] The "search" function requires a "query" argument',
                        '[Error] The "open_page" function requires either a "docid" or a "url".',
                        '[Error] The function',
                    )
                    if any(m in str(turn) for m in ERR_MARKERS):
                        agent[name].set_process_reward(i - 1, -1)
        else:
            is_finish = getattr(env, 'is_finish', False) or getattr(env, 'finish', False)
            if 'drop_fail' in process_reward:
                if not is_finish:
                    for name in branches:
                        if 'cjk' in process_reward:
                            should_drop = True
                            for i, turn in enumerate(agent[name].chat):
                                if is_weird(str(turn)):
                                    agent[name].set_process_reward(i, -2)
                                    if 'flat' in process_reward:
                                        agent[name].set_cache('reward', -1)
                                    should_drop = False
                            if should_drop:
                                agent.pop(name)
                        else:
                            agent.pop(name)  # drop all branch if not finish (overlong mask)
            # Scope check + reward
            if 'reward_scope' in process_reward:
                env.stats['scope_judge'] = 1
                for name in branches:
                    assigned_task = branch_tasks[name]
                    return_message = branch_return[name]
                    is_focus, justification = await judge_scope(assigned_task, return_message)
                    if is_focus < 0:  # scope check, skip summary turn
                        print(f'[FOCUS] Branch beyond focus: //{name}//. {justification}')
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], -0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 - 0.2)
                        env.stats['scope_judge'] = 0
                    elif is_focus > 0:
                        agent[name].set_process_reward([i for i in range(len(agent[name].chat) - 1)], 0.2)
                        if 'flat' in process_reward:
                            agent[name].set_cache('reward', 0 + 0.2)

    for name in agent if is_train else ['main']:  # in eval only return main agent
        out = await agent[name].dataproto()
        messages = agent[name].messages()
        if process_reward is not None and 'flat' in process_reward and 'reward' in agent[name].info_cache:
            out = await env.update_dataproto(out, item, messages, ('', agent[name].info_cache['reward']), reward_dict,
                                             tag=name, metrics=agent[name].get_metrics())
        else:
            out = await env.update_dataproto(out, item, messages, score, reward_dict,
                                             tag=name, metrics=agent[name].get_metrics())
        out.batch['is_overlong'] = torch.Tensor([mask_rollout])
        session_message_str = print_chat(session_message)
        out.non_tensor_batch['message_str'] = np.array([session_message_str], dtype=object)
        meta_info = f"N: {len(agent)} | {name}"
        out.non_tensor_batch['meta_info'] = np.array([meta_info], dtype=object)
        outs.append(copy.deepcopy(out))

    if max_traj is not None and len(outs) > max_traj:
        idx = [0] + sorted(random.sample(range(1, len(outs)), k=max_traj - 1))
        outs = [outs[i] for i in idx]

    try:
        res = DataProto.concat(outs)
        return res
    except Exception as e:
        breakpoint()
        return


# @register_handler("agent/fold_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
