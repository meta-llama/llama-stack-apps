# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from examples.custom_tools.ticker_data import TickerDataTool
from llama_stack import LlamaStack

from llama_stack.types import Attachment, SamplingParams, UserMessage
from llama_stack.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolCodeInterpreterToolDefinition,
    AgentConfigToolFunctionCallToolDefinition,
    AgentConfigToolMemoryToolDefinition,
    AgentConfigToolSearchToolDefinition,
    AgentConfigToolWolframAlphaToolDefinition,
)
from sdk_common.agents.event_logger import EventLogger
from sdk_common.client_utils import (
    AttachmentBehavior,
    load_api_keys_from_env,
    make_agent_config_with_custom_tools,
    QuickToolConfig,
    search_tool_defn,
)
from termcolor import cprint


async def run_main(host: str, port: int, disable_safety: bool = False):
    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )

    api_keys = load_api_keys_from_env()
    tool_definitions = [
        search_tool_defn(api_keys),
        AgentConfigToolWolframAlphaToolDefinition(
            type="wolfram_alpha",
            api_key=api_keys.wolfram_alpha,
        ),
        AgentConfigToolCodeInterpreterToolDefinition(type="code_interpreter"),
    ]

    # add ticker data as custom tool
    custom_tools = [TickerDataTool()]

    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            tool_definitions=tool_definitions,
            custom_tools=custom_tools,
            attachment_behavior="code_interpreter",
        ),
    )

    agentic_system_create_response = client.agents.create(
        agent_config=agent_config,
    )
    print(agentic_system_create_response)

    agentic_system_create_session_response = client.agents.sessions.create(
        agent_id=agentic_system_create_response.agent_id,
        session_name="test_session",
    )
    print(agentic_system_create_session_response)

    response = client.agents.turns.create(
        agent_id=agentic_system_create_response.agent_id,
        session_id=agentic_system_create_session_response.session_id,
        messages=[
            UserMessage(role="user", content="Here is a csv, can you describe it ?"),
        ],
        attachments=[
            Attachment(
                content="https://raw.githubusercontent.com/meta-llama/llama-agentic-system/main/examples/resources/inflation.csv",
                mime_type="text/csv",
            ),
        ],
        stream=True,
    )

    async for log in EventLogger().log(response):
        log.print()

    user_prompts = [
        "Which year ended with the highest inflation?",
        "What macro economic situations that led to such high inflation in that period?",
        "Plot average yearly inflation as a time series",
        "Using provided functions, get ticker data for META for the past 10 years ? plot percentage year over year growth",
        "Can you take Meta's year over year growth data and put it in the same inflation timeseries as above ?",
    ]

    for content in user_prompts:
        cprint(f"User> {content}", color="white", attrs=["bold"])

        response = client.agents.turns.create(
            agent_id=agentic_system_create_response.agent_id,
            session_id=agentic_system_create_session_response.session_id,
            messages=[
                UserMessage(content=content, role="user"),
            ],
            stream=True,
        )

        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
