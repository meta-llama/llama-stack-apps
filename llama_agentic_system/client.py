# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
from typing import Any, AsyncGenerator, List

import fire

import httpx
from llama_models.llama3_1.api.datatypes import BuiltinTool, SamplingParams, StopReason

from llama_toolchain.inference.api import Message

from llama_agentic_system.api_instance import get_agentic_system_api_instance

from .api.datatypes import (
    AgenticSystemInstanceConfig,
    AgenticSystemToolDefinition,
    AgenticSystemTurnResponseEventType,
)
from .api.endpoints import (
    AgenticSystem,
    AgenticSystemCreateRequest,
    AgenticSystemCreateResponse,
    AgenticSystemSessionCreateRequest,
    AgenticSystemSessionCreateResponse,
    AgenticSystemTurnCreateRequest,
)
from .config import AgenticSystemConfig


class AgenticSystemClient(AgenticSystem):
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def create_agentic_system(
        self, request: AgenticSystemCreateRequest
    ) -> AgenticSystemCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/create",
                data=request.json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgenticSystemCreateResponse(**response.json())

    async def create_agentic_system_session(
        self,
        request: AgenticSystemSessionCreateRequest,
    ) -> AgenticSystemSessionCreateResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/agentic_system/session/create",
                data=request.json(),
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return AgenticSystemSessionCreateResponse(**response.json())

    async def create_agentic_system_turn(
        self,
        request: AgenticSystemTurnCreateRequest,
    ) -> AsyncGenerator:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/agentic_system/turn/create",
                data=request.json(),
                headers={"Content-Type": "application/json"},
                timeout=20,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = line[len("data: ") :]
                        try:
                            yield AgenticSystemTurnResponseStreamChunk(
                                **json.loads(data)
                            )
                        except Exception as e:
                            print(data)
                            print(f"Error with parsing or validation: {e}")


EventType = AgenticSystemTurnResponseEventType


async def execute_with_custom_tools(
    system: AgenticSystem,
    system_id: str,
    session_id: str,
    messages: List[Message],
    custom_tools: List[Any],
    max_iters: int = 5,
    stream: bool = True,
) -> AsyncGenerator:
    # first create a session, or do you keep a persistent session?
    tools_dict = {t.get_name(): t for t in custom_tools}

    current_messages = messages.copy()
    n_iter = 0
    while n_iter < max_iters:
        n_iter += 1

        request = AgenticSystemTurnCreateRequest(
            system_id=system_id,
            session_id=session_id,
            messages=current_messages,
            stream=stream,
        )

        turn = None
        async for chunk in system.create_agentic_system_turn(request):
            if chunk.event.payload.event_type != EventType.turn_complete.value:
                yield chunk
            else:
                turn = chunk.event.payload.turn
                break

        message = turn.output_message
        if len(message.tool_calls) == 0:
            yield chunk
            return

        if message.stop_reason == StopReason.out_of_tokens:
            yield chunk
            return

        tool_call = message.tool_calls[0]
        if tool_call.tool_name not in tools_dict:
            m = ToolResponseMessage(
                call_id=tool_call.call_id,
                tool_name=tool_call.tool_name,
                content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
            )
            next_message = m
        else:
            tool = tools_dict[tool_call.tool_name]
            result_messages = await execute_custom_tool(tool, message)
            next_message = result_messages[0]

        yield next_message
        current_messages = [next_message]


async def execute_custom_tool(tool: Any, message: Message) -> List[Message]:
    result_messages = await tool.run([message])
    assert (
        len(result_messages) == 1
    ), f"Expected single message, got {len(result_messages)}"

    return result_messages


async def run_main(host: str, port: int):
    # client to test remote impl of agentic system
    api = await get_agentic_system_api_instance(
        AgenticSystemConfig(
            llama_distribution_url=f"http://{host}:{port}",
        )
    )

    tool_definitions = [
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.brave_search,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.wolfram_alpha,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.photogen,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.code_interpreter,
        ),
    ]

    create_request = AgenticSystemCreateRequest(
        model="Meta-Llama-3.1-8B-Instruct",
        instance_config=AgenticSystemInstanceConfig(
            instructions="You are a helpful assistant",
            sampling_params=SamplingParams(),
            available_tools=tool_definitions,
            input_shields=[],
            output_shields=[],
            quantization_config=None,
            debug_prefix_messages=[],
        ),
    )

    create_response = await api.create_agentic_system(create_request)
    print(create_response)
    # TODO: Add chat session / turn apis to test e2e


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
