# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os

import fire
from examples.custom_tools.ticker_data import TickerDataTool
from examples.custom_tools.web_search import WebSearchTool

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig


async def run_main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(f"No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    agent_config = AgentConfig(
        model="Llama3.2-3B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "brave_search",
                "engine": "brave",
                "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
            },
            {
                "type": "code_interpreter",
            },
            {
                "function_name": "get_ticker_data",
                "description": "Get yearly closing prices for a given ticker symbol",
                "parameters": {
                    "ticker_symbol": {
                        "param_type": "str",
                        "description": "The ticker symbol for which to get the data. eg. '^GSPC'",
                        "required": True,
                    },
                    "start": {
                        "param_type": "str",
                        "description": "Start date, eg. '2021-01-01'",
                        "required": True,
                    },
                    "end": {
                        "param_type": "str",
                        "description": "End date, eg. '2024-12-31'",
                        "required": True,
                    },
                },
                "type": "function_call",
            },
            {
                "function_name": "web_search",
                "description": "Search the web for a given query",
                "parameters": {
                    "query": {
                        "param_type": "str",
                        "description": "The query to search for",
                        "required": True,
                    }
                },
                "type": "function_call",
            },
        ],
        tool_choice="auto",
        tool_prompt_format="python_list",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )
    custom_tools = [TickerDataTool(), WebSearchTool(os.getenv("BRAVE_SEARCH_API_KEY"))]

    agent = Agent(client, agent_config, custom_tools)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "What was the closing price of ticker 'GOOG' for 2023 ?",
        "Who was the 42nd president of the United States?",
    ]
    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
