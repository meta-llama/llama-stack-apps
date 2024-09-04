# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403
from llama_toolchain.memory.client import MemoryClient

from multi_turn import (
    execute_turns,
    make_agent_config_with_custom_tools,
    prompt_to_turn,
    QuickToolConfig,
)


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
        MemoryBankDocument(
            document_id=f"num-{i}",
            content=URL(
                uri=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}"
            ),
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]

    memory_client = MemoryClient(f"http://{host}:{port}")
    # create a memory bank
    bank = await memory_client.create_memory_bank(
        name="test_bank",
        config=VectorMemoryBankConfig(
            bank_id="test_bank",
            embedding_model="dragon-roberta-query-2",
            chunk_size_in_tokens=512,
            overlap_size_in_tokens=64,
        ),
    )
    # insert some documents
    await memory_client.insert_documents(
        bank_id=bank.bank_id,
        documents=documents,
    )

    # now run the agentic system pointing it to the pre-populated memory bank
    agent_config = await make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(
            memory_bank_id=bank.bank_id,
            builtin_tools=[],
        ),
        disable_safety=disable_safety,
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
