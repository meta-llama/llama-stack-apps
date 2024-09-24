# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from termcolor import cprint

from llama_stack.types import *  # noqa: F403
from llama_stack.types.agent_create_params import *  # noqa: F403
from sdk_common.agents.event_logger import EventLogger
from sdk_common.client_utils import *  # noqa: F403

from .multi_turn import execute_turns, prompt_to_turn


async def run_main(host: str, port: int, disable_safety: bool = False):
    api_keys = load_api_keys_from_env()
    tool_definitions = [
        search_tool_defn(api_keys),
    ]
    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            tool_definitions=tool_definitions,
            custom_tools=[],
        ),
    )

    await execute_turns(
        agent_config=agent_config,
        custom_tools=[],
        turn_inputs=[
            prompt_to_turn(
                "I am planning a trip to Switzerland, what are the top 3 places to visit?"
            ),
            prompt_to_turn("What is so special about #1?"),
            prompt_to_turn("What other countries should I consider to club?"),
            prompt_to_turn("How many days should I plan for in each country?"),
        ],
        host=host,
        port=port,
    )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
