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
from llama_toolchain.agentic_system.utils import *  # noqa: F403

from multi_turn import execute_turns, prompt_to_turn


def main(host: str, port: int, disable_safety: bool = False):
    tool_config = QuickToolConfig(
        attachment_behavior=AttachmentBehavior.rag,
    )
    urls = [
        "memory_optimizations.rst",
        "chat.rst",
        "llama3.rst",
        "datasets.rst",
        "qat_finetune.rst",
        "lora_finetune.rst",
    ]
    attachments = [
        Attachment(
            content=URL(
                uri=f"https://raw.githubusercontent.com/pytorch/torchtune/main/docs/source/tutorials/{url}"
            ),
            mime_type="text/plain",
        )
        for i, url in enumerate(urls)
    ]
    asyncio.run(
        execute_turns(
            [
                prompt_to_turn(
                    "I am attaching some documentation for Torchtune. Help me answer questions I will ask next.",
                    attachments=attachments,
                ),
                prompt_to_turn(
                    "What are the top 5 topics that were explained? Only list succinct bullet points.",
                ),
                prompt_to_turn(
                    "Was anything related to 'Llama3' discussed, if so what?"
                ),
                prompt_to_turn(
                    "Tell me how to use LoRA",
                ),
                prompt_to_turn(
                    "What about Quantization?",
                ),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
            tool_config=tool_config,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
