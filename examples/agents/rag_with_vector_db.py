# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.tool_runtime import DocumentParam as Document
from termcolor import colored


def run_main(host: str, port: int, disable_safety: bool = False):
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
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    vector_providers = [
        provider for provider in client.providers.list() if provider.api == "vector_io"
    ]
    if not vector_providers:
        print(colored("No available vector_io providers. Exiting.", "red"))
        return

    selected_vector_provider = vector_providers[0]
    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Disabling safety.", "yellow"))
    else:
        print(f"Available shields found: {available_shields}")

    # Create a vector database instead of memory bank
    vector_db_id = "test_vector_db"
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_id=selected_vector_provider.provider_id,
    )

    # Insert documents using the RAG tool
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
    )

    available_models = [
        model.identifier for model in client.models.list() if model.model_type == "llm"
    ]
    if not available_models:
        print(colored("No available models. Exiting.", "red"))
        return

    selected_model = available_models[0]
    print(f"Using model: {selected_model}")

    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        toolgroups=[
            {
                "name": "builtin::rag",
                "args": {"vector_db_ids": [vector_db_id]},
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.",
        "Was anything related to 'Llama3' discussed, if so what?",
        "Tell me how to use LoRA",
        "What about Quantization?",
    ]

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


def main(host: str, port: int):
    run_main(host, port)


if __name__ == "__main__":
    fire.Fire(main)
