import asyncio
import json
import os
import uuid
from typing import List, Optional

import fire
import requests
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from termcolor import cprint
from tqdm import tqdm

# Initialization


def is_memory_bank_present(client, target_identifier):
    """Checks if a memory bank with the given identifier is present in the list."""
    return any(
        bank.identifier == target_identifier for bank in client.memory_banks.list()
    )


async def insert_documents_to_memory_bank(client: LlamaStackClient, docs_dir: str):
    """Inserts entire text documents from a directory into a memory bank."""
    memory_bank_id = "test_bank_0"
    providers = client.providers.list()
    provider_id = providers["memory"][0].provider_id

    memorybank_boolean = is_memory_bank_present(client, memory_bank_id)
    memorybank_list = client.memory_banks.list()
    print(memorybank_list)
    for bank in memorybank_list:
        try:
            client.memory_banks.unregister(memory_bank_id=bank.provider_resource_id)
        except Exception as e:
            print(e)

    print("after unregistration: ", client.memory_banks.list())

    if not memorybank_boolean:
        # Register a memory bank
        memory_bank = client.memory_banks.register(
            memory_bank_id=memory_bank_id,
            params={
                "embedding_model": "all-MiniLM-L6-v2",
                "chunk_size_in_tokens": 100,
                "overlap_size_in_tokens": 10,
            },
            provider_id=provider_id,
        )
        cprint(f"Memory bank registered: {memory_bank}", "green")

        # Prepare entire documents for insertion
        documents = []
        for filename in os.listdir(docs_dir):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()

                    document = Document(
                        document_id=f"{filename}",
                        content=content,
                        mime_type="text/plain",
                        metadata={"filename": filename},
                    )
                    documents.append(document)

        # Insert documents into the memory bank
        client.memory.insert(
            bank_id=memory_bank_id,
            documents=documents,
        )
        cprint(
            f"Inserted documents from {docs_dir} into memory bank '{memory_bank_id}'.",
            "green",
        )


async def run_main(host: str, port: int, docs_dir: str) -> None:
    """Main async function to register model, insert documents, and generate responses."""
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    # Insert documents to the memory bank
    await insert_documents_to_memory_bank(client, docs_dir)

    # # Model registration
    model_name = "Llama3.2-3B-Instruct"

    # Agent configuration
    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant that can answer questions based on provided documents.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": "test_bank_0", "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 512,
                "max_chunks": 5,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        enable_session_persistence=True,
    )
    agent = Agent(client, agent_config)

    session_id = agent.create_session(f"session-{uuid.uuid4()}")

    while True:
        user_input = input("User> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            cprint("Ending conversation. Goodbye!", "yellow")
            break

        cprint(f"Generating response for: {user_input}", "green")

        # Create a turn and generate the response asynchronously
        response = agent.create_turn(
            messages=[{"role": "user", "content": user_input}], session_id=session_id
        )

        # Log and display each response asynchronously
        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int, docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)
