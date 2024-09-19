# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
from typing import Optional

import fire
from llama_stack import LlamaStack
from llama_stack.types import SamplingParams, UserMessage
from llama_stack.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolCodeInterpreterToolDefinition,
    AgentConfigToolFunctionCallToolDefinition,
    AgentConfigToolSearchToolDefinition,
    AgentConfigToolWolframAlphaToolDefinition,
)
from llama_stack.types.agents import AgentsTurnStreamChunk
from llama_stack.types.tool_param_definition_param import ToolParamDefinitionParam
from termcolor import cprint


class LogEvent:
    def __init__(
        self,
        role: Optional[str] = None,
        content: str = "",
        end: str = "\n",
        color="white",
    ):
        self.role = role
        self.content = content
        self.color = color
        self.end = "\n" if end is None else end

    def __str__(self):
        if self.role is not None:
            return f"{self.role}> {self.content}"
        else:
            return f"{self.content}"

    def print(self, flush=True):
        cprint(f"{str(self)}", color=self.color, end=self.end, flush=flush)


class EventLogger:
    async def log(self, event_generator):
        previous_event_type = None
        previous_step_type = None

        for chunk in event_generator:
            if isinstance(chunk, AgenticSystemTurnStreamChunk):
                event = chunk.event
                event_type = chunk.event.payload.event_type

                if event_type in {"turn_start", "turn_complete"}:
                    # Currently not logging any turn realted info
                    continue

                step_type = chunk.event.payload.step_type
                # handle safety
                if step_type == "shield_call" and event_type == "step_complete":
                    response = event.payload.step_details.response
                    if not response.is_violation:
                        yield LogEvent(
                            role=step_type, content="No Violation", color="magenta"
                        )
                    else:
                        yield LogEvent(
                            role=step_type,
                            content=f"{response.violation_type} {response.violation_return_message}",
                            color="red",
                        )

                # handle inference
                if step_type == "inference":
                    if event_type == "step_start":
                        yield LogEvent(
                            role=step_type, content="", end="", color="yellow"
                        )
                    elif event_type == "step_progress":
                        if event.payload.tool_call_delta:
                            if isinstance(event.payload.tool_call_delta.content, str):
                                yield LogEvent(
                                    role=None,
                                    content=event.payload.tool_call_delta.content,
                                    end="",
                                    color="cyan",
                                )
                        else:
                            yield LogEvent(
                                role=None,
                                content=event.payload.text_delta_model_response,
                                end="",
                                color="yellow",
                            )
                    else:
                        # step complete
                        yield LogEvent(role=None, content="")

                # handle tool_execution
                if step_type == "tool_execution" and event_type == "step_complete":
                    details = event.payload.step_details
                    for t in details.tool_calls:
                        yield LogEvent(
                            role=step_type,
                            content=f"Tool:{t.tool_name} Args:{t.arguments}",
                            color="green",
                        )

                    for r in details.tool_responses:
                        yield LogEvent(
                            role=step_type,
                            content=f"Tool:{r.tool_name} Response:{r.content}",
                            color="green",
                        )
            else:
                yield LogEvent(chunk, color="yellow")

        preivous_event_type = event_type
        previous_step_type = step_type


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    tool_definitions = [
        AgentConfigToolSearchToolDefinition(
            type="brave_search",
            engine="brave",
            api_key="BSAd2liHqb7IjNGIpbxPRfRprAvwrbP",
        ),
        AgentConfigToolWolframAlphaToolDefinition(
            type="wolfram_alpha", api_key="78G4K7-4A299UU69P"
        ),
        AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter"),
        AgentConfigToolFunctionCallToolDefinition(
            function_name="get_boiling_point",
            description="Get the boiling point of a imaginary liquids (eg. polyjuice)",
            parameters={
                "liquid_name": ToolParamDefinitionParam(
                    param_type="str",
                    description="The name of the liquid",
                    required=True,
                ),
                "celcius": ToolParamDefinitionParam(
                    param_type="str",
                    description="Whether to return the boiling point in Celcius",
                    required=False,
                ),
            },
            type="function_call",
        ),
    ]

    agentic_system_create_response = client.agents.create(
        agent_config=AgentConfig(
            model="Meta-Llama3.1-8B-Instruct",
            instructions="You are a helpful assistant",
            sampling_params=SamplingParams(
                strategy="greedy", temperature=1.0, top_p=0.9
            ),
            tools=tool_definitions,
            tool_choices="auto",
            tool_prompt_format="function_tag",
        )
    )
    print(agentic_system_create_response)

    agentic_system_create_session_response = client.agents.sessions.create(
        agent_id=agentic_system_create_response.agent_id,
        session_name="test_session",
    )
    print(agentic_system_create_session_response)

    user_prompts = [
        "Who are you?",
        "what is the 100th prime number?",
        "Search web for who was 44th President of USA?",
        "Write code to check if a number is prime. Use that to check if 7 is prime",
        "What is the boiling point of polyjuicepotion ?",
    ]

    for content in user_prompts:
        cprint(f"User> {content}", color="white", attrs=["bold"])

        response = client.agents.turns.create(
            agent_id=agentic_system_create_response.agent_id,
            session_id=agentic_system_create_session_response.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=stream,
        )

        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
