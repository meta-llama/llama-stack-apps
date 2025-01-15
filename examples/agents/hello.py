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
from termcolor import colored


def main(host: str, port: int):
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
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        tools=(
            [
                {
                    "type": "brave_search",
                    "engine": "brave",
                    "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
                }
            ]
            if os.getenv("BRAVE_SEARCH_API_KEY")
            else []
        ),
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )
    agent = Agent(client, agent_config)
    user_prompts = [
        "Hello",
        "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools",
    ]

    session_id = agent.create_session("test-session")

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )

        for log in EventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
