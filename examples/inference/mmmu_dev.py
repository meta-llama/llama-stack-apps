# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import io

import fire
from datasets import load_dataset

from llama_stack_client import LlamaStackClient

# from llama_stack_client.lib.inference.event_logger import EventLogger
from llama_stack_client.types import SystemMessage, UserMessage
from termcolor import cprint

PROMPT_TEMPLATE = """
You are an expert in {subject} whose job is to answer questions from the user using images. First, reason about the correct answer. Then write the answer in the following format where X is exactly one of A,B,C,D: "ANSWER: X". If you are uncertain of the correct answer, guess the most likely one.
"""


def pillow_image_to_data_url(img):
    # Get the image format (PNG or JPEG)
    img_format = img.format or "PNG"
    mime_type = f"image/{img_format.lower()}"

    # Convert to base64
    buffered = io.BytesIO()
    img.save(buffered, format=img_format)
    base64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Create data URL
    return f"data:{mime_type};base64,{base64_str}"


def get_mmmu_image_url():
    ds = load_dataset(path="MMMU/MMMU", name="Accounting", split="dev")
    img = ds[1]["image_1"]
    url = pillow_image_to_data_url(img)
    print(url)
    return url


async def run_main(host: str, port: int, stream: bool = True):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )
    # img = get_mmmu_image_url()

    # 1. read dataset
    ds = load_dataset(path="MMMU/MMMU", name="Accounting", split="validation")

    # 2. loop over dataset, create user messages, and call inference
    i = 0
    for r in ds:
        print(r)
        img = r["image_1"]
        img_url = pillow_image_to_data_url(img)
        question = r["question"]
        question = question.split("<image 1>")
        system_message = SystemMessage(
            role="system",
            content=PROMPT_TEMPLATE.format(subject=r["subfield"]),
        )
        user_message = UserMessage(
            role="user",
            content=[
                question[0],
                {
                    "image": {
                        "uri": img_url,
                    }
                },
                question[1],
            ],
        )
        response = client.inference.chat_completion(
            messages=[system_message, user_message],
            model_id="meta-llama/Llama-3.2-90B-Vision-Instruct",
            sampling_params={
                "temperature": 0.0,
                "max_tokens": 4096,
            },
            stream=False,
        )
        cprint(f"Question: {r['question']}", "cyan")
        cprint(f"Response: {response}", "yellow")
        cprint(f"Answer: {r['answer']}", "cyan")
        cprint(f"Options: {r['options']}", "yellow")
        i += 1
        if i > 5:
            break

    # message = UserMessage(
    #     role="user",
    #     content=[
    #         {
    #             "image": {
    #                 "uri": img,
    #             }
    #         },
    #         "Describe what is in this image.",
    #     ],
    # )
    # cprint(f"User>{message.content}", "green")
    # response = client.inference.chat_completion(
    #     messages=[message],
    #     model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
    #     stream=stream,
    # )

    # if not stream:
    #     cprint(f"> Response: {response}", "cyan")
    # else:
    #     async for log in EventLogger().log(response):
    #         log.print()


def main(host: str, port: int, stream: bool = True):
    asyncio.run(run_main(host, port, stream))


if __name__ == "__main__":
    fire.Fire(main)
