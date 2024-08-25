# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from multi_turn import execute_turns, prompt_to_turn


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(
        execute_turns(
            [
                prompt_to_turn(
                    "I am planning a trip to Switzerland, what are the top 3 places to visit?"
                ),
                prompt_to_turn("What is so special about #1?"),
                prompt_to_turn("What other countries should I consider to club?"),
                prompt_to_turn("How many days should I plan for in each country?"),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
