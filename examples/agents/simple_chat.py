# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os

import fire
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from termcolor import colored

from .utils import check_model_is_available, get_any_available_model


def main(host: str, port: int, model_id: str | None = None):
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY is not set; please set it for this script.",
                "yellow",
            )
        )
        return

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
        provider_data={"tavily_search_api_key": os.getenv("TAVILY_SEARCH_API_KEY")},
    )

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")

    if model_id is None:
        model_id = get_any_available_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    print(f"Using model: {model_id}")

    agent = Agent(
        client,
        model=model_id,
        instructions="",
        tools=["builtin::websearch"],
        input_shields=available_shields,
        output_shields=available_shields,
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
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )

        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
