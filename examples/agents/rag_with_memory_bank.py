# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document


async def run_main(host: str, port: int, disable_safety: bool = False):
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
    providers = client.providers.list()

    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print("No available shields. Disable safety.")
    else:
        print(f"Available shields found: {available_shields}")

    # create a memory bank
    client.memory_banks.register(
        memory_bank_id="test_bank",
        params={
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
        },
        provider_id=providers["memory"][0].provider_id,
    )

    # insert some documents
    client.memory.insert(
        bank_id="test_bank",
        documents=documents,
    )
    available_models = [model.identifier for model in client.models.list()]
    selected_model = available_models[0]
    print(f"Using model: {selected_model}")

    agent_config = AgentConfig(
        model=selected_model,
        instructions="You are a helpful assistant",
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tools=[
            {
                "type": "memory",
                "memory_bank_configs": [{"bank_id": "test_bank", "type": "vector"}],
                "query_generator_config": {"type": "default", "sep": " "},
                "max_tokens_in_context": 4096,
                "max_chunks": 10,
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

        async for log in EventLogger().log(response):
            log.print()


def main(host: str, port: int):
    asyncio.run(run_main(host, port))


if __name__ == "__main__":
    fire.Fire(main)
