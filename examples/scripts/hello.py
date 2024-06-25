# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from multi_turn import prompt_to_message, run_main


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(
        run_main(
            [
                prompt_to_message("Hello"),
                prompt_to_message(
                    "Which players played in the winning team of the NBA western conference semifinals of 2024, please use tools"
                ),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
