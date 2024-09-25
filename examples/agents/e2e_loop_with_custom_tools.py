# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
import asyncio

import fire

# from llama_stack_client.agents import *  # noqa: F403
from dotenv import load_dotenv
from examples.custom_tools.ticker_data import TickerDataTool
from examples.custom_tools.web_search import WebSearchTool
from common.client_utils import *  # noqa: F403

from .multi_turn import *  # noqa: F403


def main(host: str, port: int, disable_safety: bool = False):
    api_keys = load_api_keys_from_env()
    custom_tools = [TickerDataTool(), WebSearchTool(api_keys.brave)]
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            model="Llama3.2-3B-Instruct",
            tool_config=QuickToolConfig(
                custom_tools=custom_tools,
                prompt_format="python_list",
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
                    "What was the closing price of ticker 'GOOG' for 2023 ?"
                ),
                prompt_to_turn("Who was the 42nd president of the United States?"),
            ],
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
