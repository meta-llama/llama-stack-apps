# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire
from llama_models.llama3.api.datatypes import Attachment, URL, UserMessage

from multi_turn import prompt_to_message, run_main


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(
        run_main(
            [
                UserMessage(
                    content=[
                        "here is a podcast transcript, can you summarize it",
                        Attachment(
                            url=URL(
                                uri="file://examples/resources/transcript_shorter.txt"
                            ),
                            mime_type="text/plain",
                        ),
                    ],
                ),
                prompt_to_message(
                    "What are the top 3 salient topics that were discussed ?"
                ),
                prompt_to_message("Was anything related to 'H100' discussed ?"),
                prompt_to_message(
                    "While this podcast happened in April, 2024 can you provide an update from the web on what were the key developments that have happened in the last 3 months since then ?"
                ),
                prompt_to_message(
                    "Imagine these people meet again in 1 year, what might be three good follow ups to discuss ?"
                ),
                prompt_to_message("Can you rewrite these followups in hindi ?"),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
