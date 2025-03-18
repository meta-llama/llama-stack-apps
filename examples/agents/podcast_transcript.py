# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import fire
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger, Document
from termcolor import colored


def run_main(host: str, port: int, disable_safety: bool = False):
    if "TAVILY_SEARCH_API_KEY" not in os.environ:
        print(
            colored(
                "You must set the TAVILY_SEARCH_API_KEY environment variable to use the Search tool which is required for this example.",
                "red",
            )
        )
        return

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

    selected_model = available_models[0]
    print(f"Using model: {selected_model}")

    agent = Agent(
        client,
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
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
            "Heres is a podcast transcript as attachment, can you summarize it",
            [
                Document(
                    content="https://raw.githubusercontent.com/meta-llama/llama-stack-apps/main/examples/resources/transcript_shorter.txt",
                    mime_type="text/plain",
                )
            ],
        ),
        ("What are the top 3 salient topics that were discussed ?", None),
        (
            "Was anything related to 'H100' discussed ?",
            None,
        ),
        (
            "While this podcast happened in April, 2024 can you provide an update from the web on what were the key developments that have happened in the last 3 months since then ?",
            None,
        ),
        (
            "Imagine these people meet again in 1 year, what might be three good follow ups to discuss ?",
            None,
        ),
        (
            "Can you rewrite these followups in hindi ?",
            None,
        ),
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

        for log in AgentEventLogger().log(response):
            log.print()


def main(host: str, port: int):
    run_main(host, port)


if __name__ == "__main__":
    fire.Fire(main)
