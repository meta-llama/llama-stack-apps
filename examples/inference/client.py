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
    try:
        models_response = client.models.list()  # Fetch models list from server
        available_models = [model.identifier for model in models_response]
        if not available_models:
            print("No available models in the distribution endpoint.")
            return
    except Exception as e:
        print(f"Failed to fetch models: {e}")
        return
    
    print("Available Models:")
    for idx, model_name in enumerate(available_models, start=1):
        print(f"{idx}. {model_name}")
    model_input = input(f"Select a model by number (or press Enter for default - {available_models[0]}): ")

    try:
        model_index = int(model_input) - 1 if model_input else 0
        selected_model = available_models[model_index]
    except (ValueError, IndexError):
        print("Invalid selection. Using default model.")
        selected_model = available_models[0]


    message = UserMessage(
        content="hello world, write me a 2 sentence poem about the moon", role="user"
    )
    cprint(f"User>{message.content}", "green")
    response = client.inference.chat_completion(
        messages=[message],
        model=selected_model,
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
