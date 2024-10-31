# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import AsyncGenerator, List, Optional, Union

from .custom_tools import CustomTool, Message, ToolResponseMessage

from llama_stack_client.types import *  # noqa: F403
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.agent_create_params import AgentConfig


class AgentWithCustomToolExecutor:
    def __init__(
        self,
        client: LlamaStackClient,
        agent_id: str,
        session_id: str,
        agent_config: AgentConfig,
        custom_tools: List[CustomTool],
    ):
        self.client = client
        self.agent_id = agent_id
        self.session_id = session_id
        self.agent_config = agent_config
        self.custom_tools = custom_tools

    async def execute_turn(
        self,
        messages: List[Union[UserMessage, ToolResponseMessage]],
        attachments: Optional[List[Attachment]] = None,
        max_iters: int = 5,
        stream: bool = True,
    ) -> AsyncGenerator:
        tools_dict = {t.get_name(): t for t in self.custom_tools}

        current_messages = messages.copy()
        n_iter = 0

        while n_iter < max_iters:
            n_iter += 1
            response = self.client.agents.turn.create(
                agent_id=self.agent_id,
                session_id=self.session_id,
                messages=current_messages,
                attachments=attachments,
                stream=True,
            )
            turn = None
            for chunk in response:
                if chunk.event.payload.event_type != "turn_complete":
                    yield chunk
                else:
                    turn = chunk.event.payload.turn

            message = turn.output_message
            if len(message.tool_calls) == 0:
                yield chunk
                return

            if message.stop_reason == "out_of_tokens":
                yield chunk
                return

            tool_call = message.tool_calls[0]
            if tool_call.tool_name not in tools_dict:
                m = ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
                    role="ipython",
                )
                next_message = m
            else:
                tool = tools_dict[tool_call.tool_name]
                result_messages = await execute_custom_tool(tool, message)
                next_message = result_messages[0]

            yield next_message
            current_messages = [next_message]


async def execute_custom_tool(tool: CustomTool, message: Message) -> List[Message]:
    result_messages = await tool.run([message])
    assert (
        len(result_messages) == 1
    ), f"Expected single message, got {len(result_messages)}"

    return result_messages
