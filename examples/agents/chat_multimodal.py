# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import base64
from pathlib import Path

import fire
from llama_stack_client import Agent, LlamaStackClient
from termcolor import colored

from .utils import check_model_is_available

THIS_DIR = Path(__file__).parent


def main(host: str, port: int, model_id: str | None = None):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    if model_id is None:
        print(
            colored(
                "No model id provided. Specify a model that supports vision.", "red"
            )
        )
        return
    else:
        if not check_model_is_available(client, model_id):
            print(
                colored(
                    "Model not available. Specify a model that supports vision.", "red"
                )
            )
            return

    agent = Agent(
        client,
        model=model_id,
        instructions="",
    )

    files = [
        THIS_DIR / "resources" / "dog.png",
        THIS_DIR / "resources" / "pasta.jpeg",
    ]
    encoded_files = []
    for file in files:
        with open(file, "rb") as image_file:
            base64_data = base64.b64encode(image_file.read()).decode("utf-8")
            encoded_files.append(base64_data)

    prompts = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "data": encoded_files[0],
                    },
                },
                {
                    "type": "text",
                    "text": "Write a haiku about this image",
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": {
                        "data": encoded_files[1],
                    },
                },
                {
                    "type": "text",
                    "text": "Now update the haiku to include the second image",
                },
            ],
        },
    ]

    session_id = agent.create_session("test-session")
    for prompt in prompts:
        response = agent.create_turn(
            messages=[prompt],
            session_id=session_id,
            stream=False,
        )
        print(colored("model>", "green"), response.output_message.content)


if __name__ == "__main__":
    fire.Fire(main)
