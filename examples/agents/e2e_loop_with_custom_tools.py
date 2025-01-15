# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

from examples.custom_tools.ticker_data import TickerDataTool
from examples.custom_tools.web_search import WebSearchTool


async def run_main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list().data]
    if not available_shields:
        print("No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    available_models = [
        model.identifier
        for model in client.models.list().data
        if model.model_type == "llm"
    ]
    supported_models = [x for x in available_models if "3.2" in x and "Vision" not in x]
    if not supported_models:
        raise ValueError(
            "No supported models found. Make sure to have a Llama 3.2 model."
        )
    else:
        selected_model = supported_models[0]
        print(f"Using model: {selected_model}")
    client_tools = [
        TickerDataTool(),
        WebSearchTool(os.getenv("BRAVE_SEARCH_API_KEY")),
    ]
    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        tools=[
            "builtin::code_interpreter",
        ],
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        tool_choice="auto",
        tool_prompt_format="python_list",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config, client_tools)
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

        for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
