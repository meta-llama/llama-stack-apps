# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from multi_turn import (
    BuiltinTool,
    execute_turns,
    make_agent_config_with_custom_tools,
    prompt_to_turn,
    QuickToolConfig,
)


def main(host: str, port: int, disable_safety: bool = False):
    custom_tools = []
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                builtin_tools=[
                    BuiltinTool.brave_search,
                ],
            ),
            disable_safety=disable_safety,
        )
    )
    asyncio.run(
        execute_turns(
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
    )


if __name__ == "__main__":
    fire.Fire(main)
