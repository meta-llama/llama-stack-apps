# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types import Document
from termcolor import colored
from uuid import uuid4
from rich.pretty import pprint

def run_main(host: str, port: int, disable_safety: bool = False):


    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return

    selected_model = available_models[0]
    selected_model = "meta-llama/Llama-4-17B-Llama-API"
    print(f"Using model: {selected_model}")

    response = client.inference.chat_completion(
        model_id=selected_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a haiku about coding"},
        ],
        stream=False,
    )
    pprint(f"Response: {response}")
    available_shields =[]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")
    from termcolor import cprint

    agent = Agent(
        client, 
        model="meta-llama/Llama-3.1-8B-Instruct",
        instructions="You are a helpful assistant. Use websearch tool to help answer questions.",
        tools=["builtin::websearch"],
    )
    user_prompts = [
        "Hello",
        "Which teams played in the NBA western conference finals of 2024",
    ]

    session_id = agent.create_session("test-session")
    for prompt in user_prompts:
        cprint(f"User> {prompt}", "green")
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            session_id=session_id,
        )
        for log in AgentEventLogger().log(response):
            log.print()
 


def main(host: str, port: int):
    run_main(host, port)


if __name__ == "__main__":
    fire.Fire(main)
