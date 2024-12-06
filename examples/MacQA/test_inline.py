import asyncio
import json
import os
import re
import uuid
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, Generator, List, Optional

import gradio as gr
import requests
from llama_stack.apis.memory_banks.memory_banks import VectorMemoryBankParams
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger

# pip install aiosqlite ollama faiss autoevals opentelemetry-exporter-otlp-proto-http
from llama_stack_client.lib.direct.direct import LlamaStackDirectClient
from llama_stack_client.types import SystemMessage, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


async def main():
    os.environ["INFERENCE_MODEL"] = "meta-llama/Llama-3.2-1B-Instruct"
    client = await LlamaStackDirectClient.from_template("ollama")
    await client.initialize()
    response = await client.models.list()
    print(response)
    model_name = response[0].identifier
    response = await client.inference.chat_completion(
        messages=[
            SystemMessage(content="You are a friendly assistant.", role="system"),
            UserMessage(
                content="hello world, write me a 2 sentence poem about the moon",
                role="user",
            ),
        ],
        model_id=model_name,
        stream=False,
    )
    print("\nChat completion response:")
    print(response, type(response))
    print("\n\n")
    await setup_memory_bank(client, "macqa_bank")


async def setup_memory_bank(client, memory_bank_id):
    """Set up the memory bank if it doesn't exist."""
    providers = await client.providers.list()
    provider_id = providers["memory"][0].provider_id
    memory_banks = await client.memory_banks.list()
    print(f"Memory banks: {memory_banks}")

    # Check if memory bank exists by identifier
    if any(bank.identifier == memory_bank_id for bank in memory_banks):
        print(f"Memory bank '{memory_bank_id}' exists.")
    else:
        print(f"Memory bank '{memory_bank_id}' does not exist. Creating...")
        await client.memory_banks.register(
            memory_bank_id=memory_bank_id,
            params=VectorMemoryBankParams(
                embedding_model="all-MiniLM-L6-v2",
                chunk_size_in_tokens=100,
                overlap_size_in_tokens=10,
            ),
            #             params={
            #     "embedding_model": "all-MiniLM-L6-v2",
            #     "chunk_size_in_tokens": 100,
            #     "overlap_size_in_tokens": 10,
            # },
            provider_id=provider_id,
        )
        print(f"Memory bank registered.")


asyncio.run(main())
