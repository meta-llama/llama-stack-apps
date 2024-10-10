# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from common.client_utils import *  # noqa: F403

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import Attachment
from llama_stack_client.types.memory_insert_params import Document

from .multi_turn import execute_turns, prompt_to_turn


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
    # create a memory bank
    client.memory_banks.register(
        memory_bank={
            "identifier": "test_bank",
            "embedding_model": "all-MiniLM-L6-v2",
            "chunk_size_in_tokens": 512,
            "overlap_size_in_tokens": 64,
            "provider_id": "meta-reference",
        }
    )

    # insert some documents
    client.memory.insert(
        bank_id="test_bank",
        documents=documents,
    )

    # now run the agentic system pointing it to the pre-populated memory bank
    agent_config = await make_agent_config_with_custom_tools(
        model="Llama3.1-8B-Instruct",
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            # enable memory for RAG behavior, provide appropriate bank_id
            memory_bank_id="test_bank",
            attachment_behavior="rag",
        ),
    )

    await execute_turns(
        agent_config=agent_config,
        custom_tools=[],
        turn_inputs=[
            prompt_to_turn(
                "What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.",
            ),
            prompt_to_turn("Was anything related to 'Llama3' discussed, if so what?"),
            prompt_to_turn(
                "Tell me how to use LoRA",
            ),
            prompt_to_turn(
                "What about Quantization?",
            ),
        ],
        host=host,
        port=port,
    )


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(run_main(host, port, disable_safety))


if __name__ == "__main__":
    fire.Fire(main)
