# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid
from typing import List

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import ToolResponseMessage, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn_create_response import (
    AgentTurnResponseStreamChunk,
)

from .base_env import BaseEnv
from .base_tool import BaseTool


class TauAgent:
    def __init__(
        self,
        env: BaseEnv,
        tools: List[BaseTool],
        agent_config: AgentConfig,
    ):
        self.client = LlamaStackClient(
            base_url="http://localhost:5000",
        )
        self.env = env
        self.tools = tools
        self.agent_config = agent_config
        self.agent = Agent(self.client, self.agent_config, custom_tools=self.tools)
        self.session_id = self.agent.create_session(f"test-session-{uuid.uuid4()}")

    def reset(self) -> str:
        # Clear the session
        self.session_id = self.agent.create_session(f"test-session-{uuid.uuid4()}")
        return self.session_id

    def step(self, message: UserMessage, verbose: bool = True) -> str:
        response = self.agent.create_turn(
            messages=[message],
            session_id=self.session_id,
        )
        chunks = [chunk for chunk in response]
        last_chunk = chunks[-1]

        if verbose:
            for log in EventLogger().log(chunks):
                log.print()

        while isinstance(last_chunk, ToolResponseMessage):
            response = self.agent.create_turn(
                messages=[last_chunk],
                session_id=self.session_id,
            )
            chunks = [chunk for chunk in response]
            last_chunk = chunks[-1]
            if verbose:
                for log in EventLogger().log(chunks):
                    log.print()

        assert isinstance(last_chunk, AgentTurnResponseStreamChunk)
        return last_chunk.event.payload.turn.output_message
