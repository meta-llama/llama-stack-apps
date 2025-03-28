# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from uuid import uuid4

import fire
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient, RAGDocument
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
    documents = [
        RAGDocument(
            document_id=f"num-{i}",
            content=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}",
            mime_type="text/plain",
            metadata={},
        )
        for i, url in enumerate(urls)
    ]

    client = LlamaStackClient(base_url=f"http://{host}:{port}")
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
    vector_db_id = f"test_vector_db_{uuid4()}"
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

    print(f"Using model: {model_id}")

    agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use knowledge_search tool to gather information needed to answer questions. Answer succintly.",
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
        tools=[
            {
                "name": "builtin::rag/knowledge_search",
                "args": {"vector_db_ids": [vector_db_id]},
            }
        ],
        input_shields=available_shields if available_shields else [],
        output_shields=available_shields if available_shields else [],
    )
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
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
        print(f"User> {prompt}")
        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
