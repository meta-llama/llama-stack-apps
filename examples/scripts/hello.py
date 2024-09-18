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
    execute_turns,
    load_api_keys_from_env,
    make_agent_config_with_custom_tools,
    prompt_to_turn,
    QuickToolConfig,
    search_tool_defn,
)


def main(host: str, port: int, disable_safety: bool = False):
    custom_tools = []
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                builtin_tools=[
                    search_tool_defn(load_api_keys_from_env()),
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
                prompt_to_turn("Hello"),
                prompt_to_turn(
                    "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools"
                ),
            ],
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
