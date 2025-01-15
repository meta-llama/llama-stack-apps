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
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn_create_params import Document
from termcolor import colored


async def run_main(host: str, port: int, disable_safety: bool = False):
    if "BRAVE_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: BRAVE_SEARCH_API_KEY is not set; will not use Search tool.",
                "yellow",
            )
        )

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list().data]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")
    available_models = [
        model.identifier
        for model in client.models.list().data
        if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return
    else:
        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

    agent_config = AgentConfig(
        model=selected_model,
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        instructions="You are a helpful assistant",
        tools=(
            (
                ["builtin::websearch"]
                if os.getenv("BRAVE_SEARCH_API_KEY")
                or os.getenv("TAVILY_SEARCH_API_KEY")
                else []
            )
            + ["builtin::code_interpreter"]
        ),
        tool_choice="required",
        tool_prompt_format="json",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        (
            "Here is a csv, can you describe it ?",
            [
                Document(
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
            documents=prompt[1],
            session_id=session_id,
        )

        for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
