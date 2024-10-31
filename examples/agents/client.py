# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from common.client_utils import *  # noqa: F403

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger

from llama_stack_client.types import *  # noqa: F403
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint

from .multi_turn import *  # noqa: F403


class Agent:
    def __init__(self, host: str, port: int):
        self.client = LlamaStackClient(
            base_url=f"http://{host}:{port}",
        )

    def create_agent(self, agent_config: AgentConfig):
        agentic_system_create_response = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = agentic_system_create_response.agent_id
        agentic_system_create_session_response = self.client.agents.session.create(
            agent_id=agentic_system_create_response.agent_id,
            session_name="test_session",
        )
        self.session_id = agentic_system_create_session_response.session_id

    async def execute_turn(self, content: str):
        response = self.client.agents.turn.create(
            agent_id=self.agent_id,
            session_id=self.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=True,
        )
        for chunk in response:
            if chunk.event.payload.event_type != "turn_complete":
                yield chunk


async def run_main(host: str, port: int, disable_safety: bool = False):
    tool_definitions = [
        AgentConfigToolSearchToolDefinition(
            type="brave_search", engine="brave", api_key="YOUR_API_KEY"
        )
    ]

    agent_config = AgentConfig(
        model="Llama3.1-8B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(strategy="greedy", temperature=1.0, top_p=0.9),
        tools=tool_definitions,
        tool_choice="auto",
        tool_prompt_format="function_tag",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    agent = Agent(host=host, port=port)
    agent.create_agent(agent_config)

    user_prompts = [
        "I am planning a trip to Switzerland, what are the top 3 places to visit?",
        "What is so special about #1?",
        "What other countries should I consider to club?",
        "What is the capital of France?",
    ]

    iterator = agent.execute_turn(content="What is the capital of France?")

    for prompt in user_prompts:
        cprint(f"User> {prompt}", color="white", attrs=["bold"])
        response = agent.execute_turn(content=prompt)
        async for log in EventLogger().log(response):
            if log is not None:
                log.print()


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
