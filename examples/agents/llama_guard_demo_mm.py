# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

import fire
from llama_models.llama3.api import *  # noqa: F403
from examples.interior_design_assistant.utils import data_url_from_image  # noqa: F403
from llama_stack_client import LlamaStackClient


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

    for img_path in [unsafe_example_img_path, safe_example_img_path]:

        print(f"Running on input : prompt: {prompt}, image: {img_path}")
        img_media = ImageMedia(image=URL(uri=data_url_from_image(img_path)))
        for message in [UserMessage(content=[prompt, img_media], role="user")]:
            response = client.safety.run_shield(
                messages=[message],
                shield_type="llama_guard",
                params={},
            )

            print(response)


if __name__ == "__main__":
    fire.Fire(main)
