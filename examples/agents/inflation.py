# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from llama_stack_client.types import Attachment, SamplingParams, UserMessage
from llama_stack_client.types.agent_create_params import *  # noqa: F403
from common.client_utils import *  # noqa: F403
from termcolor import cprint

from .multi_turn import execute_turns, prompt_to_turn


async def run_main(host: str, port: int, disable_safety: bool = False):
    api_keys = load_api_keys_from_env()
    tool_definitions = [
        search_tool_defn(api_keys),
        # Adding code_interpreter enables file analysis
        AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter"),
    ]

    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            tool_definitions=tool_definitions,
            custom_tools=[],
            attachment_behavior="code_interpreter",
        ),
    )

    await execute_turns(
        agent_config=agent_config,
        custom_tools=[],
        turn_inputs=[
            prompt_to_turn(
                "Here is a csv, can you describe it ?",
                attachments=[
                    Attachment(
                        content="https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/inflation.csv",
                        mime_type="text/csv",
                    ),
                ],
            ),
            prompt_to_turn("Which year ended with the highest inflation ?"),
            prompt_to_turn(
                "What macro economic situations that led to such high inflation in that period?"
            ),
            prompt_to_turn("Plot average yearly inflation as a time series"),
        ],
        host=host,
        port=port,
    )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
