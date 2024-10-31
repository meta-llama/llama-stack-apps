# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import base64
import mimetypes
import os

import fire
import pandas

from datasets import load_dataset

from llama_stack_client import LlamaStackClient
from termcolor import cprint

from .client import data_url_from_file

INPUT_QUERY_PROMPT = """
Given the following question and candidate answers, choose the best answer.
Question: {question}
A. {A}
B. {B}
C. {C}
D. {D}

Your response should end with "The best answer is [the_answer_letter]." where the [the_answer_letter] is a letter from the provided choices.

Let's think step by step.
"""


def process_hf_dataset() -> str:
    # return saved file path
    data = load_dataset(
        "meta-llama/Llama-3.2-1B-Instruct-evals",
        name="Llama-3.2-1B-Instruct-evals__mmlu__details",
        split="latest",
    )
    # transform it into accceptable format for evals
    # <input_query, chat_completion_input, expected_output>
    input_query = []
    chat_completion_input = []
    expected_output = []

    for x in data:
        query = INPUT_QUERY_PROMPT.format(
            question=x["input_question"],
            **x["input_choice_list"],
        )
        input_query.append(query)
        chat_completion_input.append(
            [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        )
        expected_output.append(x["input_correct_responses"][0])

    transformed_data = {
        "input_query": input_query,
        "chat_completion_input": chat_completion_input,
        "expected_answer": expected_output,
    }

    df = pandas.DataFrame(transformed_data)
    df.to_csv("mmlu.csv", index=False)

    print(df.head())


async def run_main(host: str, port: int, file_path: str):
    # process_hf_dataset()

    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    providers = client.providers.list()

    dataset_url = data_url_from_file(file_path)

    client.datasets.register(
        dataset_def={
            "identifier": "eval-mmlu",
            "provider_id": providers["datasetio"][0].provider_id,
            "url": {"uri": dataset_url},
            "dataset_schema": {
                "expected_answer": {"type": "string"},
                "input_query": {"type": "string"},
                "chat_completion_input": {"type": "string"},
            },
        }
    )

    datasets_list_response = client.datasets.list()
    cprint([x.identifier for x in datasets_list_response], "cyan")

    # test eval with individual rows
    rows_paginated = client.datasetio.get_rows_paginated(
        dataset_id="eval-mmlu",
        rows_in_page=3,
        page_token=None,
        filter_condition=None,
    )
    # print(rows_paginated)

    eval_candidate = {
        "type": "model",
        "model": "Llama3.2-1B-Instruct",
        "sampling_params": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_tokens": 0,
            "repetition_penalty": 1.0,
        },
    }
    eval_rows = client.eval.evaluate(
        input_rows=rows_paginated.rows,
        candidate=eval_candidate,
        scoring_functions=[
            "meta-reference::subset_of",
            # "meta-reference::llm_as_judge_8b_correctness",
        ],
    )
    cprint(eval_rows, "green")


def main(host: str, port: int, file_path: str):
    asyncio.run(run_main(host, port, file_path))


if __name__ == "__main__":
    fire.Fire(main)
