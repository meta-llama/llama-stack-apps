# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import uuid

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig

from ..envs.database import RetailDatabaseEnv

from .configs.wiki import WIKI
from .tools.calculate import CalculateTool


class RetailAgent:
    def __init__(self):
        self.client = LlamaStackClient(
            base_url="http://localhost:5000",
        )
        self.database = RetailDatabaseEnv()
        self.tools = [
            CalculateTool(self.database),
        ]
        self.agent_config = AgentConfig(
            model="meta-llama/Llama-3.1-405B-Instruct-FP8",
            instructions=WIKI,
            tools=[tool.get_tool_definition() for tool in self.tools],
            sampling_params={
                "strategy": "greedy",
                "temperature": 1.0,
                "top_p": 0.9,
            },
            tool_choice="auto",
            tool_prompt_format="json",
            input_shields=[],
            output_shields=[],
            enable_session_persistence=False,
        )
        self.agent = Agent(self.client, self.agent_config, custom_tools=self.tools)
        self.session_id = self.agent.create_session(f"test-session-{uuid.uuid4()}")

    def step(self, user_prompt: str) -> str:
        response = self.agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            session_id=self.session_id,
        )
        for log in EventLogger().log(response):
            log.print()
