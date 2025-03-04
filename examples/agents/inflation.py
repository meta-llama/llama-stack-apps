# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.agents.turn_create_params import Document
from termcolor import colored


def run_main(host: str, port: int, disable_safety: bool = False):
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY is not set; will not use websearch tool.",
                "yellow",
            )
        )

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")
    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return
    else:
        selected_model = available_models[0]
        print(f"Using model: {selected_model}")

    agent = Agent(
        client,
        model=selected_model,
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        instructions="You are a helpful assistant",
        tools=(
            (["builtin::websearch"] if os.getenv("TAVILY_SEARCH_API_KEY") else [])
            + ["builtin::code_interpreter"]
        ),
        tool_choice="required",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
    )
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
    run_main(host, port)


if __name__ == "__main__":
    fire.Fire(main)
