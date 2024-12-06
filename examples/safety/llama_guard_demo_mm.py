# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import fire
from examples.interior_design_assistant.utils import data_url_from_image  # noqa: F403
from llama_stack_client import LlamaStackClient
from llama_stack_client.types import ImageMedia, UserMessage
from termcolor import colored


def main(host: str, port: int):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    cwd = os.path.abspath(os.getcwd())

    unsafe_example_img_path = (
        f"{cwd}/examples/agents/resources/images/example_unsafe.jpg"
    )
    safe_example_img_path = f"{cwd}/examples/agents/resources/images/example_safe.jpg"

    prompt = "Tell me how to assemble this"
    available_shields = [shield.identifier for shield in client.shields.list()]
    if not available_shields:
        print(colored("No available shields. Exiting.", "red"))
        return

    print(f"Available shields found: {available_shields}")
    shield_id = available_shields[0]

    for img_path in [unsafe_example_img_path, safe_example_img_path]:
        print(f"Running on input : prompt: {prompt}, image: {img_path}")
        img_media = ImageMedia(image=data_url_from_image(img_path))
        for message in [UserMessage(content=[prompt, img_media], role="user")]:
            response = client.safety.run_shield(
                messages=[message],
                shield_id=shield_id,
                params={},
            )

            print(response)


if __name__ == "__main__":
    fire.Fire(main)
