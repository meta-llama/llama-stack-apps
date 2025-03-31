# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import fire
from termcolor import colored

from examples.client_tools.ticker_data import get_ticker_data
from examples.client_tools.web_search import WebSearchTool
from examples.client_tools.calculator import calculator

from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger

from .utils import check_model_is_available, get_any_available_model


def main(host: str, port: int, model_id: str | None = None):
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    api_key = ""
    engine = "tavily"
    if "TAVILY_SEARCH_API_KEY" in os.environ:
        api_key = os.getenv("TAVILY_SEARCH_API_KEY")
    elif "BRAVE_SEARCH_API_KEY" in os.environ:
        api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        engine = "brave"
    else:
        print(
            colored(
                "Warning: TAVILY_SEARCH_API_KEY or BRAVE_SEARCH_API_KEY is not set; Web search will not work",
                "yellow",
            )
        )

    if model_id is None:
        model_id = get_any_available_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use the tools you have access to for providing relevant answers.",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        tools=[
            calculator,
            get_ticker_data,
            # Note: While you can also use "builtin::websearch" as a tool,
            # this example shows how to use a client side custom web search tool.
            WebSearchTool(engine, api_key),
        ],
    )
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "What was the closing price of Google stock (ticker symbol GOOG) for 2023 ?",
        "Who was the 42nd president of the United States?",
        "What is 40+30?",
    ]
    for prompt in user_prompts:
        print(colored(f"User> {prompt}", "cyan"))
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )

        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
