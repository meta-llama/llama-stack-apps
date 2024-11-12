# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import base64
import mimetypes
import os

import fire
import pandas as pd

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document
from termcolor import cprint
from tqdm import tqdm


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


def build_index(client: LlamaStackClient, file_dir: str, bank_id: str) -> str:
    """Build a memory bank from a directory of pdf files"""
    # 1. create memory bank
    providers = client.providers.list()
    client.memory_banks.register(
        memory_bank={
            "identifier": bank_id,
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
            "provider_id": providers["memory"][0].provider_id,
        }
    )

    # 2. load pdf,text,md from directory as raw text
    paths = []
    documents = []
    for filename in os.listdir(file_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(file_dir, filename)
            paths.append(file_path)

            documents.append(
                Document(
                    document_id=os.path.basename(file_path),
                    content=data_url_from_file(file_path),
                    mime_type="application/pdf",
                )
            )
        elif filename.endswith(".txt") or filename.endswith(".md"):
            file_path = os.path.join(file_dir, filename)
            paths.append(file_path)
            documents.append(
                Document(
                    document_id=os.path.basename(file_path),
                    content=data_url_from_file(file_path),
                    mime_type="text/plain",
                )
            )

    # insert some documents
    client.memory.insert(bank_id=bank_id, documents=documents)
    print(f"Inserted {len(documents)} documents into bank: {bank_id}")
    # TODO: add a way to check if the bank is created successfully
    return bank_id


async def get_response_row(agent: Agent, input_query: str) -> str:
    # single turn, each prompt is a new session
    session_id = agent.create_session(f"session-{input_query}")
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": input_query,
            }
        ],
        session_id=session_id,
    )

    async for chunk in response:
        event = chunk.event
        event_type = event.payload.event_type
        if event_type == "turn_complete":
            return event.payload.turn.output_message.content


async def run_main(host: str, port: int, docs_dir: str):
    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    bank_id = "rag_agent_docs"
    build_index(client, docs_dir, bank_id)
    print(f"Created bank: {bank_id}")
    models_response = client.models.list()
    print(f"Found {len(models_response)} models", models_response)
    model_name = None
    for model in models_response:
        if not model_name and model.identifier.endswith("Instruct"):
            model_name = model.llama_model
            print(f"Use model: {model_name}")
    assert model_name is not None, "No model found"
    agent_config = AgentConfig(
        model=model_name,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            # TODO: save the bank to local file and load it from local file
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": bank_id, "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 50,
            }
        ],
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    agent = Agent(client, agent_config)

    # load dataset and generate responses for the RAG agent
    user_prompts = ["What methods are best for finetuning llama models?"]

    llamastack_generated_responses = []

    for prompt in tqdm(user_prompts):
        print(f"Generating response for: {prompt}")
        try:
            generated_response = await get_response_row(agent, prompt)
            llamastack_generated_responses.append(generated_response)
        except Exception as e:
            print(f"Error generating response for {prompt}: {e}")
            llamastack_generated_responses.append(None)
    # TODO: make this multi-turn instead of single turn
    for response in llamastack_generated_responses:
        print(response)


def main(host: str, port: int, docs_dir: str):
    asyncio.run(run_main(host, port, docs_dir))


if __name__ == "__main__":
    fire.Fire(main)
