# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import asyncio

import fire

from llama_models.llama3_1.api.datatypes import *  # noqa: F403
from custom_tools.ticker_data import TickerDataTool

from multi_turn import prompt_to_message, run_main


def main(host: str, port: int, disable_safety: bool = False):
    asyncio.run(
        run_main(
            [
                UserMessage(
                    content=[
                        "Here is a csv, can you describe it ?",
                        Attachment(
                            url=URL(uri="file://examples/resources/inflation.csv"),
                            mime_type="text/csv",
                        ),
                    ],
                ),
                prompt_to_message("Which year ended with the highest inflation ?"),
                prompt_to_message(
                    "What macro economic situations that led to such high inflation in that period?"
                ),
                prompt_to_message("Plot average yearly inflation as a time series"),
                prompt_to_message(
                    "Using provided functions, get ticker data for META for the past 10 years ? plot percentage year over year growth"
                ),
                prompt_to_message(
                    "Can you take Meta's year over year growth data and put it in the same inflation timeseries as above ?"
                ),
            ],
            host=host,
            port=port,
            disable_safety=disable_safety,
            custom_tools=[TickerDataTool()],
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
