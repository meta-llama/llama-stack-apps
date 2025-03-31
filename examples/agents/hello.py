# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from typing import Optional

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from termcolor import colored


def main(host: str, port: int, model_id: Optional[str] = None):
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY is not set; please set it for this script.",
                "yellow",
            )
        )
        exit()

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
        provider_data={"tavily_search_api_key": os.getenv("TAVILY_SEARCH_API_KEY")},
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")

    llm_inference_provider_ids = [
        p.provider_id
        for p in client.providers.list()
        if p.api == "inference" and p.provider_type != "sentence-transformers"
    ]

    if model_id is not None:
        client.models.register(
            model_id=model_id,
            model_type="llm",  # model_type
            provider_id=llm_inference_provider_ids[0],
            provider_model_id=model_id,
        )

    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm"
        and "405B" not in model.identifier
        and "405b" not in model.identifier
        and "guard" not in model.identifier
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return
    elif model_id is not None:
        selected_model = model_id
    else:
        selected_model = available_models[0]
    print(f"Using model: {selected_model}")

    agent = Agent(
        client,
        model=selected_model,
        instructions="",
        tools=["builtin::websearch"],
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )
    user_prompts = [
        "Hello",
        "Search web for which players played in the winning team of the NBA western conference semifinals of 2024",
    ]

    session_id = agent.create_session("test-session")
    for prompt in user_prompts:
        print(f"User> {prompt}")
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
