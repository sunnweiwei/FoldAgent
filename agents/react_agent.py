import os
import asyncio

from verl import DataProto
from .utils import Agent, select_env, TaskContext, run_action, AgentLoopOutput, AgentLoopMetrics
from .prompts import create_chat


async def process_item(
        item: DataProto,
        context: TaskContext,
) -> AgentLoopOutput:
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

    workflow = item.non_tensor_batch['extra_info'][0].get('workflow', None) or getattr(config.plugin, "workflow",
                                                                                       "search")
    user_prompt = create_chat(env.instance_info['problem_statement'], workflow, item)
    max_turn = getattr(config.plugin, 'max_turn', 64) if config.plugin else 64

    llm_client = context.llm_client
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

    out = await agent.get_data()
    agent_reward = score[1]
    out = AgentLoopOutput(
        prompt_ids=out['prompt_ids'],
        response_ids=out['response_ids'],
        response_mask=out['response_mask'],
        response_logprobs=out['response_logprobs'],
        multi_modal_data={},
        metrics=AgentLoopMetrics(),
        reward_score=agent_reward,
        num_turns=out['num_turns'],
    )
    return out


# @register_handler("agent/react_agent")
# class ReActAgent(AsyncAgent):
#     async def __call__(self, item: DataProto, context: TaskContext, **kwargs):
#         return await process_single_batch(item, context)
