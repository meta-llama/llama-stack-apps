# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio
import os
from typing import Optional
from urllib.parse import urlparse

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig


async def run_main(
    host: str,
    port: int,
    use_https: bool = False,
    cert_path: Optional[str] = None,
):
    # Construct the base URL with the appropriate protocol
    protocol = "https" if use_https else "http"
    base_url = f"{protocol}://{host}:{port}"

    # Configure client with SSL certificate if provided
    client_kwargs = {"base_url": base_url}
    if use_https and cert_path:
        client_kwargs["verify"] = cert_path

    client = LlamaStackClient(**client_kwargs)

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(f"No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    agent_config = AgentConfig(
        model="Llama3.1-8B-Instruct",
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
            }
        ],
        tool_choice="auto",
        tool_prompt_format="function_tag",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "I am planning a trip to Switzerland, what are the top 3 places to visit?",
        "What is so special about #1?",
        "What other countries should I consider to club?",
        "What is the capital of France?",
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


def main(
    host: str,
    port: int,
    use_https: bool = False,
    cert_path: Optional[str] = None,
):
    asyncio.run(run_main(host, port, use_https, cert_path))


if __name__ == "__main__":
    fire.Fire(main)
