# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import abc
import uuid
from typing import Optional

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import cprint


class BaseUser(abc.ABC):
    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError


class HumanUser(BaseUser):
    def reset(self, instruction: str) -> str:
        cprint(f"Resetting User...\n{instruction}", "grey")
        return input("\n> ")

    def step(self, content: str) -> str:
        return input("> ")


class SimulatedUser(BaseUser):
    def __init__(self, instruction: str) -> None:
        self.client = LlamaStackClient(
            base_url="http://localhost:5000",
        )
        system_prompt = self.build_system_prompt(instruction)
        self.agent_config = AgentConfig(
            model="meta-llama/Llama-3.1-405B-Instruct-FP8",
            instructions=system_prompt,
            tools=[],
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
        self.agent = Agent(self.client, self.agent_config)
        self.session_id = self.agent.create_session(f"test-session-{uuid.uuid4()}")

    def reset(self, instruction: str) -> str:
        cprint(f"Resetting User...\n{instruction}\n", "grey")
        self.agent_config["instructions"] = self.build_system_prompt(instruction)
        self.agent = Agent(self.client, self.agent_config)
        self.session_id = self.agent.create_session(f"test-session-{uuid.uuid4()}")

        response = self.agent.create_turn(
            messages=[
                {"role": "user", "content": "Hi! How can I help you today?"},
            ],
            session_id=self.session_id,
        )
        chunks = [chunk for chunk in response]
        return chunks[-1].event.payload.turn.output_message.content

    def step(self, content: str) -> str:
        response = self.agent.create_turn(
            messages=[{"role": "user", "content": content}],
            session_id=self.session_id,
        )
        chunks = [chunk for chunk in response]
        return chunks[-1].event.payload.turn.output_message.content

    def build_system_prompt(self, instruction: Optional[str]) -> str:
        instruction_display = (
            ("\n\nInstruction: " + instruction + "\n")
            if instruction is not None
            else ""
        )
        return f"""You are a user interacting with an agent. {instruction_display}
        Rules:
        - Just generate one line at a time to simulate the user's message.
        - Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
        - Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
        - If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
        - Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
        - Try to make the conversation as natural as possible, and stick to the personalities in the instruction.
        """
