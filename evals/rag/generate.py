# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os

import fire
import pandas as pd

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.memory_insert_params import Document
from tqdm import tqdm

from ..util import agent_bulk_generate, data_url_from_file

from .config import AGENT_CONFIG, MEMORY_BANK_ID, MEMORY_BANK_PARAMS


def build_index(
    client: LlamaStackClient, file_dir: str, bank_id: str, bank_params: dict
) -> None:
    """Build a memory bank from a directory of pdf files"""
    # 1. create memory bank
    providers = client.providers.list()
    client.memory_banks.register(
        memory_bank_id=bank_id,
        params={
            **bank_params,
            "provider_id": providers["memory"][0].provider_id,
        },
    )

    # 2. load pdfs from directory as raw text
    paths = []
    for filename in os.listdir(file_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(file_dir, filename)
            paths.append(file_path)

    for p in paths:
        documents = [
            Document(
                document_id=os.path.basename(p),
                content=data_url_from_file(p),
                mime_type="application/pdf",
            )
        ]
        # insert some documents
        client.memory.insert(bank_id=bank_id, documents=documents)


async def run_main(host: str, port: int, docs_dir: str, input_file_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
        provider_data={
            "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
        },
    )

    build_index(client, docs_dir, MEMORY_BANK_ID, MEMORY_BANK_PARAMS)
    agent = Agent(client, AGENT_CONFIG)
    await agent_bulk_generate(agent, input_file_path)


def main(host: str, port: int, docs_dir: str, input_file_path: str):
    asyncio.run(run_main(host, port, docs_dir, input_file_path))


if __name__ == "__main__":
    fire.Fire(main)
