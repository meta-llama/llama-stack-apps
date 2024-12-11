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

    message = UserMessage(
        role="user",
        content=[
            {
                "image": {
                    "uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                }
            },
            "Describe what is in this image.",
        ],
    )
    cprint(f"User>{message.content}", "green")
    response = client.inference.chat_completion(
        messages=[message],
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        stream=stream,
    )

    if not stream:
        cprint(f"> Response: {response}", "cyan")
    else:
        for log in EventLogger().log(response):
            log.print()

    # query models endpoint
    models_response = client.models.list()
    print(models_response)


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
