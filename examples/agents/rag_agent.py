# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from uuid import uuid4

import fire
import time
from llama_stack_client import Agent, AgentEventLogger, LlamaStackClient, RAGDocument
from termcolor import colored

from .utils import check_model_is_available, get_any_available_model


def main(
    host: str,
    port: int,
    model_id: str | None = None,
):
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

    if model_id is None:
        model_id = get_any_available_model(client)
        if model_id is None:
            return
    else:
        if not check_model_is_available(client, model_id):
            return

    print(f"Using model: {model_id}")

    vector_providers = [
        provider for provider in client.providers.list() if provider.api == "vector_io"
    ]
    if not vector_providers:
        print(colored("No available vector_io providers. Exiting.", "red"))
        return

    selected_vector_provider = vector_providers[0]

    # Create a vector database
    vector_db_id = f"test_vector_db_{uuid4()}"
    client.vector_dbs.register(
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
        provider_id=selected_vector_provider.provider_id,
    )

    # Insert documents using the RAG tool
    start_time = time.time()
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
    )
    end_time = time.time()
    print(colored(f"Inserted documents in {end_time - start_time:.2f}s", "cyan"))

    agent = Agent(
        client,
        model=model_id,
        instructions="You are a helpful assistant. Use knowledge_search tool to gather information needed to answer questions. Answer succintly.",
        tools=[
            {
                "name": "builtin::rag/knowledge_search",
                "args": {"vector_db_ids": [vector_db_id]},
            }
        ],
        # Optionally can set sampling params
        sampling_params={
            "strategy": {"type": "top_p", "temperature": 1.0, "top_p": 0.9},
        },
    )
    session_id = agent.create_session("test-session")
    print(f"Created session_id={session_id} for Agent({agent.agent_id})")

    user_prompts = [
        "Is anything related to 'Llama3' mentioned, if so what?",
        "Tell me how to use LoRA",
        "What about Quantization?",
    ]

    for prompt in user_prompts:
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt}],
            session_id=session_id,
        )
        print(colored(f"User> {prompt}", "blue"))
        for log in AgentEventLogger().log(response):
            log.print()


if __name__ == "__main__":
    fire.Fire(main)
