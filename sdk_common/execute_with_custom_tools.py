# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import AsyncGenerator, List, Optional

from .custom_tools import CustomTool

from llama_stack.types import *  # noqa: F403
from llama_stack import LlamaStack
from llama_stack.types.agent_create_params import AgentConfig


class AgentWithCustomToolExecutor:
    def __init__(
        self,
        client: LlamaStack,
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
        messages: List[UserMessage],
        attachments: Optional[List[Attachment]] = None,
        max_iters: int = 1,
        stream: bool = True,
    ) -> AsyncGenerator:
        # tools_dict = {t.get_name(): t for t in self.custom_tools}

        current_messages = messages.copy()
        n_iter = 0
        while n_iter < max_iters:
            n_iter += 1
            response = self.client.agents.turns.create(
                agent_id=self.agent_id,
                session_id=self.session_id,
                messages=current_messages,
                attachments=attachments,
                stream=True,
            )
            turn = None
            for chunk in response:
                yield chunk
