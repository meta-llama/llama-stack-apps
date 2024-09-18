# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from llama_stack.apis.agents import *  # noqa: F403
from examples.custom_tools.ticker_data import TickerDataTool

from multi_turn import (
    AttachmentBehavior,
    execute_turns,
    load_api_keys_from_env,
    make_agent_config_with_custom_tools,
    prompt_to_turn,
    QuickToolConfig,
    search_tool_defn,
    WolframAlphaToolDefinition,
)


def main(host: str, port: int, disable_safety: bool = False):
    api_keys = load_api_keys_from_env()
    custom_tools = [TickerDataTool()]
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                builtin_tools=[
                    search_tool_defn(api_keys),
                    WolframAlphaToolDefinition(api_key=api_keys.wolfram_alpha),
                ],
                attachment_behavior=AttachmentBehavior.code_interpreter,
                custom_tools=custom_tools,
            ),
            disable_safety=disable_safety,
        )
    )
    asyncio.run(
        execute_turns(
            agent_config=agent_config,
            custom_tools=custom_tools,
            turn_inputs=[
                prompt_to_turn(
                    "Here is a csv, can you describe it ?",
                    attachments=[
                        Attachment(
                            content=URL(
                                uri="https://raw.githubusercontent.com/meta-llama/llama-agentic-system/main/examples/resources/inflation.csv",
                            ),
                            mime_type="text/csv",
                        ),
                    ],
                ),
                prompt_to_turn("Which year ended with the highest inflation ?"),
                prompt_to_turn(
                    "What macro economic situations that led to such high inflation in that period?"
                ),
                prompt_to_turn("Plot average yearly inflation as a time series"),
                prompt_to_turn(
                    "Using provided functions, get ticker data for META for the past 10 years ? plot percentage year over year growth"
                ),
                prompt_to_turn(
                    "Can you take Meta's year over year growth data and put it in the same inflation timeseries as above ?"
                ),
            ],
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
