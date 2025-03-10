# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import os

from llama_stack.distribution.library_client import AsyncLlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import AsyncAgent
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.types.agent_create_params import AgentConfig
from rich.pretty import pprint


@client_tool
async def async_calculator(x: float, y: float, operation: str) -> dict:
    """Simple calculator tool that performs basic math operations.

    :param x: First number to perform operation on
    :param y: Second number to perform operation on
    :param operation: Mathematical operation to perform ('add', 'subtract', 'multiply', 'divide')
    :returns: Dictionary containing success status and result or error message
    """
    print(f"Calculator called with: x={x}, y={y}, operation={operation}")
    try:
        if operation == "add":
            result = float(x) + float(y)
        elif operation == "subtract":
            result = float(x) - float(y)
        elif operation == "multiply":
            result = float(x) * float(y)
        elif operation == "divide":
            if float(y) == 0:
                return {"success": False, "error": "Cannot divide by zero"}
            result = float(x) / float(y)
        else:
            return {"success": False, "error": "Invalid operation"}

        print(f"Calculator result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        print(f"Calculator error: {str(e)}")
        return {"success": False, "error": str(e)}


async def main():
    client = AsyncLlamaStackAsLibraryClient(
        "together",
        provider_data={"tavily_search_api_key": os.environ["TAVILY_SEARCH_API_KEY"]},
    )
    _ = await client.initialize()

    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    client_tools = [async_calculator]
    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant",
        toolgroups=["builtin::websearch"],
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    async_agent = AsyncAgent(client, agent_config, client_tools)

    session_id = await async_agent.create_session("test-session")
    print("AGENT_ID", async_agent.agent_id)
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
    pprint(turn)

    turn2 = await async_agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": "What is 2 + 2?",
            }
        ],
        session_id=session_id,
        stream=True,
    )

    async for chunk in turn2:
        pprint(chunk)


if __name__ == "__main__":
    asyncio.run(main())
