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

from llama_stack import LlamaStack
from llama_stack.types import Attachment
from llama_stack.types.memory_insert_params import Document

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

    client = LlamaStack(
        base_url=f"http://{host}:{port}",
    )
    # create a memory bank
    bank = client.memory.create(
        body={
            "name": "test_bank",
            "config": {
                "type": "vector",
                "bank_id": "test_bank",
                "embedding_model": "dragon-roberta-query-2",
                "chunk_size_in_tokens": 512,
                "overlap_size_in_tokens": 64,
            },
        },
    )

    # insert some documents
    client.memory.insert(
        bank_id=bank["bank_id"],
        documents=documents,
    )

    # now run the agentic system pointing it to the pre-populated memory bank
    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=QuickToolConfig(
            memory_bank_id=bank["bank_id"],
            tool_definitions=[],
            custom_tools=[],
            attachment_behavior="rag",
        ),
    )
    print(agent_config)

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
