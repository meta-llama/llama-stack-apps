# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import fire
from llama_stack_client import Agent, AgentEventLogger, Document, LlamaStackClient
from termcolor import colored


def main(host: str, port: int, model_id: str, disable_safety: bool = False):
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]

    attachments = [
        Document(
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return

    if model_id not in available_models:
        available_models_str = "\n".join(available_models)
        print(
            f"Model `{model_id}` not found. Available models:\n\n{available_models_str}\n"
        )
        print(colored("Exiting.", "red"))
        return

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")

    selected_model = model_id
    print(f"Using model: {selected_model}")

    agent = Agent(
        client,
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        tools=["builtin::rag/knowledge_search"],
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
    )
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        (
            "I am attaching some documentation for Torchtune. Help me answer questions I will ask next by using the knowledge_search tool.",
            attachments,
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
        print(f"User> {prompt[0]}")

        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
