# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Attachment
from llama_stack_client.types.agent_create_params import AgentConfig


async def run_main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    input_shields = [] if disable_safety else ["llama_guard"]
    output_shields = [] if disable_safety else ["llama_guard"]

    agent_config = AgentConfig(
        model="Llama3.1-8B-Instruct",
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "brave_search",
                "engine": "brave",
                "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
            },
            {
                "type": "code_interpreter",
            },
        ],
        tool_choice="required",
        tool_prompt_format="json",
        input_shields=input_shields,
        output_shields=output_shields,
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        (
            "Here is a csv, can you describe it ?",
            [
                Attachment(
                    content="https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/inflation.csv",
                    mime_type="test/csv",
                )
            ],
        ),
        ("Which year ended with the highest inflation ?", None),
        (
            "What macro economic situations that led to such high inflation in that period?",
            None,
        ),
        ("Plot average yearly inflation as a time series", None),
    ]

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt[0],
                }
            ],
            attachments=prompt[1],
            session_id=session_id,
        )

        async for log in EventLogger().log(response):
            log.print()

    # api_keys = load_api_keys_from_env()
    # tool_definitions = [
    #     search_tool_defn(api_keys),
    #     # Adding code_interpreter enables file analysis
    #     AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter"),
    # ]

    # agent_config = await make_agent_config_with_custom_tools(
    #     disable_safety=disable_safety,
    #     tool_config=QuickToolConfig(
    #         tool_definitions=tool_definitions,
    #         custom_tools=[],
    #         attachment_behavior="code_interpreter",
    #     ),
    # )

    # print(agent_config)

    # await execute_turns(
    #     agent_config=agent_config,
    #     custom_tools=[],
    #     turn_inputs=[
    #         prompt_to_turn(
    #             "Here is a csv, can you describe it ?",
    #             attachments=[
    #                 Attachment(
    #                     content="https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/inflation.csv",
    #                     mime_type="text/csv",
    #                 ),
    #             ],
    #         ),
    #         prompt_to_turn("Which year ended with the highest inflation ?"),
    #         prompt_to_turn(
    #             "What macro economic situations that led to such high inflation in that period?"
    #         ),
    #         prompt_to_turn("Plot average yearly inflation as a time series"),
    #     ],
    #     host=host,
    #     port=port,
    # )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
