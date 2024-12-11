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
from rich.prompt import Prompt
from termcolor import cprint


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )
    models_response = client.models.list()  # Fetch models list from server
    available_models = [model.identifier for model in models_response]

    if not available_models:
        cprint("No available models in the distribution endpoint.", "red")
        return

    print("Available Models:")
    for idx, model_name in enumerate(available_models, start=1):
        print(f"{idx}. {model_name}")

    selected_model_idx = Prompt.ask(
        f"Select a model by number (or press Enter for default - {available_models[0]})",
        choices=[str(i) for i in range(1, len(available_models) + 1)],
        default="1",
    )
    selected_model = available_models[int(selected_model_idx) - 1]

    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    cprint(f"User>{message.content}", "green")
    response = client.inference.chat_completion(
        messages=[message],
        model_id=selected_model,
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
