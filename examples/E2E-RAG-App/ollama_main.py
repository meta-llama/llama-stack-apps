import asyncio
import json
import os
import uuid
from typing import List, Optional

import fire
import requests
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from termcolor import cprint
from tqdm import tqdm

# Initialization
load_dotenv()


async def insert_documents_to_memory_bank(client: LlamaStackClient, docs_dir: str):
    """Inserts entire text documents from a directory into a memory bank."""
    memory_bank_id = "test_bank_3"
    providers = client.providers.list()
    provider_id = providers["memory"][0].provider_id

    # Register a memory bank
    memory_bank = client.memory_banks.register(
        memory_bank_id=memory_bank_id,
        params={
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
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

    # Model registration
    model_name = "Llama3.2-3B-Instruct"
    response = requests.post(
        f"http://{host}:{port}/models/register",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model_id": model_name,
                "provider_model_id": None,
                "provider_id": "ollama",
                # "provider_id": "inline::meta-reference-0",
                "metadata": None,
            }
        ),
    )
    cprint(f"Model registration status: {response.status_code}", "blue")

    # Agent configuration
    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant that can answer questions based on provided documents.",
        sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": "test_bank_3", "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        enable_session_persistence=True,
    )
    agent = Agent(client, agent_config)

    while True:
        user_input = input("User> ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            cprint("Ending conversation. Goodbye!", "yellow")
            break

        message = {"role": "user", "content": user_input}
        response = agent.create_turn(
            messages=[
                {
                    "role": "user",
                    "content": message,
                }
            ],
            session_id=session_id,
        )

        async for log in EventLogger().log(response):
            log.print()


# Run the chat loop in a Jupyter Notebook cell using await


def main(host: str, port: int, docs_dir: str) -> None:
    """Entry point for the script."""
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)
