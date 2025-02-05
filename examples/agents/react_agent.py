# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import uuid
from typing import Any, Dict, List, Optional, Union

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.client_tool import ClientTool
from llama_stack_client.lib.agents.react.output_parser import ReActOutputParser
from llama_stack_client.lib.agents.react.prompts import (
    DEFAULT_REACT_AGENT_SYSTEM_PROMPT_TEMPLATE,
)

from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.shared.tool_response_message import ToolResponseMessage
from llama_stack_client.types.shared.user_message import UserMessage
from llama_stack_client.types.tool_def_param import Parameter

from pydantic import BaseModel
from rich.pretty import pprint


class Action(BaseModel):
    tool_name: str
    tool_params: Dict[str, Any]


class ReActOutput(BaseModel):
    thought: str
    action: Optional[Action] = None
    answer: Optional[str] = None


class SearchTool(ClientTool):

    def get_name(self) -> str:
        return "search"

    def get_description(self) -> str:
        return """
        Search the web for the given query.
        """

    def get_params_definition(self) -> Dict[str, Parameter]:
        return {
            "query": Parameter(
                name="query",
                parameter_type="str",
                description="The query to use for querying the internet",
                required=True,
            )
        }

    def run(
        self, messages: List[Union[UserMessage, ToolResponseMessage]]
    ) -> List[Union[UserMessage, ToolResponseMessage]]:
        dummy_response = """
            torchtune is a PyTorch library for easily authoring, finetuning and experimenting with LLMs.

            torchtune provides:

            PyTorch implementations of popular LLMs from Llama, Gemma, Mistral, Phi, and Qwen model families
            Hackable training recipes for full finetuning, LoRA, QLoRA, DPO, PPO, QAT, knowledge distillation, and more
            Out-of-the-box memory efficiency, performance improvements, and scaling with the latest PyTorch APIs
            YAML configs for easily configuring training, evaluation, quantization or inference recipes
            Built-in support for many popular dataset formats and prompt templates
        """
        return [
            ToolResponseMessage(
                call_id=messages[0].tool_calls[0].call_id,
                tool_name=self.get_name(),
                content=dummy_response,
                role="tool",
            )
        ]


def main():
    client = LlamaStackClient(
        base_url="http://localhost:8321",
    )

    model = "meta-llama/Llama-3.1-8B-Instruct"

    client_tools = [
        SearchTool(),
    ]

    tool_names = ", ".join([tool.get_name() for tool in client_tools])
    tool_descriptions = "\n".join(
        [f"- {tool.get_name()}: {tool.get_description()}" for tool in client_tools]
    )
    instruction = DEFAULT_REACT_AGENT_SYSTEM_PROMPT_TEMPLATE.replace(
        "<<tool_names>>", tool_names
    ).replace("<<tool_descriptions>>", tool_descriptions)

    agent_config = AgentConfig(
        model=model,
        instructions=instruction,
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        client_tools=[
            client_tool.get_tool_definition() for client_tool in client_tools
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
        # response_format={
        #     "type": "json_schema",
        #     "json_schema": ReActOutput.model_json_schema(),
        # },
    )

    agent = Agent(client, agent_config, client_tools, output_parser=ReActOutputParser())

    session_id = agent.create_session(f"ttest-session-{uuid.uuid4().hex}")

    response = agent.create_turn(
        messages=[
            {"role": "user", "content": "What model families does torchtune support?"}
        ],
        session_id=session_id,
        stream=False,
    )
    pprint(response)


if __name__ == "__main__":
    fire.Fire(main)
