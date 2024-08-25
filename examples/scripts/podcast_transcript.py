# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio
from pathlib import Path

import fire
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.agentic_system.utils import *  # noqa: F403

from multi_turn import execute_turns, prompt_to_turn

SCRIPTS = Path(__file__).parent
EXAMPLES = SCRIPTS.parent


def main(host: str, port: int, disable_safety: bool = False):
    tool_config = QuickToolConfig(
        attachment_behavior=AttachmentBehavior.code_interpreter,
    )
    transcript_path = EXAMPLES / "resources/transcript_shorter.txt"
    asyncio.run(
        execute_turns(
            [
                prompt_to_turn(
                    "here is a podcast transcript, can you summarize it",
                    attachments=[
                        Attachment(
                            content=URL(
                                uri=f"file://{str(transcript_path.resolve())}",
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
            disable_safety=disable_safety,
            tool_config=tool_config,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
