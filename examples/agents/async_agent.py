# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os

from llama_stack.distribution.library_client import AsyncLlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import AsyncAgent
from llama_stack_client.types.agent_create_params import AgentConfig


async def main():
    client = AsyncLlamaStackAsLibraryClient(
        "together",
        provider_data={"tavily_search_api_key": os.environ["TAVILY_SEARCH_API_KEY"]},
    )
    _ = await client.initialize()

    model_id = "meta-llama/Llama-3.3-70B-Instruct"

    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant",
        toolgroups=["builtin::websearch"],
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    async_agent = AsyncAgent(client, agent_config)
    await async_agent.initialize()

    session_id = await async_agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({async_agent.agent_id})")

    turn = await async_agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "What is the capital of France?",
            }
        ],
        session_id=session_id,
        stream=False,
    )
    print(turn)
    # async for chunk in turn:
    #     print(chunk)


if __name__ == "__main__":
    asyncio.run(main())
