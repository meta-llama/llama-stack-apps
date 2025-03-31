# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import fire
from llama_stack_client import Agent, AgentEventLogger, Document, LlamaStackClient
from termcolor import colored

from .utils import check_model_is_available, get_any_available_model


def main(host: str, port: int, model_id: str | None = None):
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]

    documents = [
        Document(
            content={
                "uri": f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            },
            mime_type="text/plain",
        )
        for _, url in enumerate(urls)
    ]

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    if model_id is None:
        model_id = get_any_available_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    print(f"Using model: {model_id}")

    selected_model = model_id
    print(f"Using model: {selected_model}")

    agent = Agent(
        client,
        model=selected_model,
        instructions="You are a helpful assistant",
    )
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        (
            "I am attaching some documentation for Torchtune to ask some questions.",
            documents,
        ),
        (
            "What are the top 5 topics that were explained? Only list succinct bullet points.",
            None,
        ),
        (
            "Was anything related to 'Llama3' discussed, if so what?",
            None,
        ),
        (
            "Tell me how to use LoRA",
            None,
        ),
        (
            "What about Quantization?",
            None,
        ),
    ]

    for prompt, documents in user_prompts:
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            documents=documents,
            session_id=session_id,
        )
        print(colored(f"User> {prompt}", "blue"))
        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
