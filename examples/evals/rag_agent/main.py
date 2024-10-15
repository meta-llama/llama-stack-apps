# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import fire
from .agent import *
from llama_stack_client.lib.agents.event_logger import EventLogger
from termcolor import cprint
import json
import pandas

import pickle
from pathlib import Path


async def test_agent(host: str, port: int, file_dir: str):
    agent = await get_rag_agent(host, port, file_dir)
    user_prompts = [
        "What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.",
    ]

    for prompt in user_prompts:
        cprint(f"User> {prompt}", color="white", attrs=["bold"])
        response = agent.execute_turn(content=prompt)
        async for log in EventLogger().log(response):
            if log is not None:
                log.print()


def main(host: str, port: int, file_dir: str, eval_dataset: str = "", no_cache: bool = False):
    asyncio.run(test_agent(host, port, file_dir))


if __name__ == "__main__":
    fire.Fire(main)
