import logging
import os
from typing import Any, Union

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl import DataProto
from agents.fold_agent import process_item
from agents.utils import CallLLM, TaskContext

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("fold_agent")
class FoldAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True

        logger.info("Initializing FoldAgentLoop class")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.config = config

    async def run(
        self, sampling_params: dict[str, Any], **kwargs
    ) -> Union[AgentLoopOutput, list[AgentLoopOutput]]:
        item = DataProto.from_dict(non_tensors=kwargs)

        llm_client = CallLLM(
            url=self.server_manager,
            tokenizer=self.tokenizer,
            config=self.config.actor_rollout_ref.rollout,
            loop=self.loop,
        )
        context = TaskContext(
            config=self.config,
            global_step=kwargs.get('global_step', 0),
            llm_client=llm_client,
            is_train=kwargs.get('is_train', True),
            tokenizer=self.tokenizer,
        )
        rollout_results = await process_item(item, context)

        return rollout_results


def main():
    """Entry point for training - runs VERL's main PPO trainer with fold_agent registered."""
    from verl.trainer.main_ppo import main as verl_main
    verl_main()


if __name__ == "__main__":
    main()
