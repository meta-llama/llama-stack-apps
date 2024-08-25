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
from examples.custom_tools.ticker_data import TickerDataTool

from multi_turn import execute_turns, prompt_to_turn

SCRIPTS = Path(__file__).parent
EXAMPLES = SCRIPTS.parent


def main(host: str, port: int, disable_safety: bool = False):
    tool_config = QuickToolConfig(
        attachment_behavior=AttachmentBehavior.code_interpreter,
        custom_tools=[TickerDataTool()],
    )
    inflation_path = EXAMPLES / "resources/inflation.csv"
    asyncio.run(
        execute_turns(
            [
                prompt_to_turn(
                    "Here is a csv, can you describe it ?",
                    attachments=[
                        Attachment(
                            content=URL(
                                uri=f"file://{str(inflation_path.resolve())}",
                            ),
                            mime_type="text/csv",
                        ),
                    ],
                ),
                prompt_to_turn("Which year ended with the highest inflation ?"),
                prompt_to_turn(
                    "What macro economic situations that led to such high inflation in that period?"
                ),
                prompt_to_turn("Plot average yearly inflation as a time series"),
                prompt_to_turn(
                    "Using provided functions, get ticker data for META for the past 10 years ? plot percentage year over year growth"
                ),
                prompt_to_turn(
                    "Can you take Meta's year over year growth data and put it in the same inflation timeseries as above ?"
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
