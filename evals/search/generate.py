# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import os

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent

from ..util import agent_bulk_generate

from .config import AGENT_CONFIG


async def run_main(host: str, port: int, input_file_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
        provider_data={
            "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
        },
    )
    agent = Agent(client, AGENT_CONFIG)
    await agent_bulk_generate(agent, input_file_path)


def main(host: str, port: int, input_file_path: str):
    asyncio.run(run_main(host, port, input_file_path))


if __name__ == "__main__":
    fire.Fire(main)
