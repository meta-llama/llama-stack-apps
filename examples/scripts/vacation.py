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
                prompt_to_message(
                    "I am planning a trip to Switzerland, what are the top 3 places to visit?"
                ),
                prompt_to_message("What is so special about #1?"),
                prompt_to_message("What other countries should I consider to club?"),
                prompt_to_message("How many days should I plan for in each country?"),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
