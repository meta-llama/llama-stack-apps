# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from common.client_utils import *  # noqa: F403

from llama_stack import LlamaStack

from llama_stack.types import SamplingParams, UserMessage
from llama_stack.types.agent_create_params import AgentConfig
from termcolor import cprint

from .multi_turn import execute_turns, prompt_to_turn


async def run_main(host: str, port: int, disable_safety: bool = False):
    tool_definitions = [search_tool_defn(load_api_keys_from_env())]
    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(tool_definitions=tool_definitions),
    )

    await execute_turns(
        agent_config=agent_config,
        custom_tools=[],
        turn_inputs=[
            prompt_to_turn(
                "Hello",
            ),
            prompt_to_turn(
                "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools"
            ),
        ],
        host=host,
        port=port,
    )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
