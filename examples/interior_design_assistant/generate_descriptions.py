# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import glob
import os
from pathlib import Path

import fire

from examples.interior_design_assistant.utils import (
    create_single_turn,
    data_url_from_image,
)

from llama_stack_client import LlamaStackClient
from llama_stack_client.types import agent_create_params


MODEL = "Llama3.2-11B-Vision-Instruct"


def main(host: str, port: int, image_dir: str, output_dir: str):
    paths = []
    patterns = ["*.png", "*.jpeg", "*.jpg", "*.webp"]
    for p in patterns:
        for i, file_ in enumerate(glob.glob(os.path.join(image_dir, p))):
            paths.append(os.path.basename(file_))

    cfg = agent_create_params.AgentConfig(
        model=MODEL,
        instructions="",
        sampling_params=agent_create_params.SamplingParams(
            strategy="greedy", temperature=0.0
        ),
        enable_session_persistence=False,
    )
    # check if output dir exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    memory_dir = Path(output_dir)

    client = LlamaStackClient(base_url=f"http://{host}:{port}")

    paths = sorted(paths)
    for p in paths:
        full_path = os.path.join(image_dir, p)
        message = {
            "role": "user",
            "content": [
                {"image": {"uri": data_url_from_image(full_path)}},
                "Describe the design, style, color, material and other aspects of the fireplace in this photo. Respond in one paragraph, no bullet points or sections.",
            ],
        }

        response = create_single_turn(client, cfg, [message])
        response += "\n\n"
        response += f"<uri>{p}</uri>"

        with open(memory_dir / f"{p}.txt", "w") as f:
            f.write(response)

        print(f"Finsihed processing {p}")


if __name__ == "__main__":
    fire.Fire(main)
