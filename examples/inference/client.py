# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import UserMessage
from termcolor import cprint


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    client.models.register(
        model={
            "identifier": "Llama3.1-8B-Instruct",
            "llama_model": "Llama3.1-8B-Instruct",
            "provider_id": "meta-reference-0",
            "metadata": {},
        }
    )

    message = {
        "role": "user",
        "content": "hello world, write me a 2 sentence poem about the moon",
    }

    cprint(f"User>{message.content}", "green")
    response = client.inference.chat_completion(
        messages=[message],
        model="Llama3.1-8B-Instruct",
        stream=stream,
    )

    if not stream:
        cprint(f"> Response: {response}", "cyan")
    else:
        async for log in EventLogger().log(response):
            log.print()

    # query models endpoint
    models_response = client.models.list()
    print(models_response)


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
