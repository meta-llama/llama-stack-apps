# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger as AgentEventLogger
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


def main(config_path: str):
    client = LlamaStackAsLibraryClient(config_path)
    client.initialize()

    models = client.models.list()
    print("\nModels:")
    for model in models:
        print(model)

    if not models:
        print("No models found, skipping chat completion test")
        return

    for model in models:
        if model.model_type == "llm":
            model_id = model.identifier

    print(f"\nUsing model: {model_id}")
    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=False,
    )
    print("\nChat completion response (non-stream):")
    print(response)

    response = client.inference.chat_completion(
        messages=[UserMessage(content="What is the capital of France?", role="user")],
        model_id=model_id,
        stream=True,
    )

    print("\nChat completion response (stream):")
    for log in EventLogger().log(response):
        log.print()

    print("\nAgent test:")
    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=(
            [
                {
                    "type": "brave_search",
                    "engine": "brave",
                    "api_key": os.getenv("BRAVE_SEARCH_API_KEY"),
                }
            ]
            if os.getenv("BRAVE_SEARCH_API_KEY")
            else []
        ),
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )
    agent = Agent(client, agent_config)
    user_prompts = [
        "Hello",
        "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools",
    ]

    session_id = agent.create_session("test-session")

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

        for log in AgentEventLogger().log(response):
            log.print()

    # memory test
    print("\nMemory test:")
    memory_bank_id = "mem_bank"
    setup_memory_bank(client, memory_bank_id)
    load_documents(client, memory_bank_id, "example_data")
    agent_config = AgentConfig(
        model=model_id,
        instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": memory_bank_id, "type": "vector"}],
                "max_tokens_in_context": 300,
                "max_chunks": 5,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        enable_session_persistence=True,
    )
    mem_agent = Agent(client, agent_config)
    session_id = mem_agent.create_session(f"session-memory-{memory_bank_id}")
    message = "What is llama 3.2?"
    response = mem_agent.create_turn(
        messages=[{"role": "user", "content": message}],
        session_id=session_id,
    )
    for log in AgentEventLogger().log(response):
        log.print()


def load_documents(client, memory_bank_id, docs_dir):
    """Load documents from the specified directory into memory bank."""
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith((".txt", ".md")):
            file_path = os.path.join(docs_dir, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                document = Document(
                    document_id=filename,
                    content=content,
                    mime_type="text/plain",
                    metadata={"filename": filename},
                )
                documents.append(document)

    if documents:
        client.memory.insert(
            bank_id=memory_bank_id,
            documents=documents,
        )
        print(f"Loaded {len(documents)} documents from {docs_dir}")


def setup_memory_bank(client, memory_bank_id):
    """Set up the memory bank if it doesn't exist."""
    providers = client.providers.list()
    provider_id = providers["memory"][0].provider_id
    memory_banks = client.memory_banks.list()
    print(f"Memory banks: {memory_banks}")

    # Check if memory bank exists by identifier
    if any(bank.identifier == memory_bank_id for bank in memory_banks):
        print(f"Memory bank '{memory_bank_id}' exists.")
    else:
        print(f"Memory bank '{memory_bank_id}' does not exist. Creating...")
        client.memory_banks.register(
            memory_bank_id=memory_bank_id,
            params={
                "memory_bank_type": "vector",
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 100,
                "overlap_size_in_tokens": 10,
            },
            provider_id=provider_id,
        )
        print(f"Memory bank registered.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the config YAML file")
    args = parser.parse_args()
    main(args.config_path)
