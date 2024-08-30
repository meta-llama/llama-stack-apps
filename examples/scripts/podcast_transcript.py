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

from multi_turn import (
    AttachmentBehavior,
    BuiltinTool,
    execute_turns,
    make_agent_config_with_custom_tools,
    prompt_to_turn,
    QuickToolConfig,
)


def main(host: str, port: int, disable_safety: bool = False):
    agent_config = asyncio.run(
        make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                builtin_tools=[
                    BuiltinTool.brave_search,
                    BuiltinTool.wolfram_alpha,
                ],
                attachment_behavior=AttachmentBehavior.code_interpreter,
            ),
            disable_safety=disable_safety,
        )
    )
    transcript_path = "https://raw.githubusercontent.com/meta-llama/llama-agentic-system/main/examples/resources/transcript_shorter.txt"
    asyncio.run(
        execute_turns(
            agent_config=agent_config,
            custom_tools=[],
            turn_inputs=[
                prompt_to_turn(
                    "here is a podcast transcript, can you summarize it",
                    attachments=[
                        Attachment(
                            content=URL(
                                uri=transcript_path,
                            ),
                            mime_type="text/plain",
                        ),
                    ],
                ),
                prompt_to_turn(
                    "What are the top 3 salient topics that were discussed ?"
                ),
                prompt_to_turn("Was anything related to 'H100' discussed ?"),
                prompt_to_turn(
                    "While this podcast happened in April, 2024 can you provide an update from the web on what were the key developments that have happened in the last 3 months since then ?"
                ),
                prompt_to_turn(
                    "Imagine these people meet again in 1 year, what might be three good follow ups to discuss ?"
                ),
                prompt_to_turn("Can you rewrite these followups in hindi ?"),
            ],
            host=host,
            port=port,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
