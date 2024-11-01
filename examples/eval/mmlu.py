# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
from pathlib import Path
from typing import Optional

import fire
import pandas

from datasets import load_dataset

from llama_stack_client import LlamaStackClient
from termcolor import cprint

from .client import data_url_from_file

INPUT_QUERY_PROMPT = """
Given the following question and candidate answers, choose the best answer. 

Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD.
"""

INPUT_QUERY_PROMPT_STRICT = """
Given the following question and candidate answers, choose the best answer. 

Question: {question}

A) {A}
B) {B}
C) {C}
D) {D}

Your response should only contain a single letter '$LETTER' (without quotes) where LETTER is one of ABCD.
"""


def process_hf_dataset(dataset, dataset_name, dataset_path, strict):
    data = load_dataset(
        dataset,
        name=dataset_name,
        split="latest",
    )

    # transform it into accceptable format for evals
    # <input_query, chat_completion_input, expected_output>
    input_query = []
    chat_completion_input = []
    expected_output = []

    for x in data:
        query_prompt = INPUT_QUERY_PROMPT_STRICT if strict else INPUT_QUERY_PROMPT
        query = query_prompt.format(
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
    df.to_csv(dataset_path, index=False)
    print(f"Saved dataset at {dataset_path}!")


def run_eval(host: str, port: int, file_path: str, strict: bool):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    providers = client.providers.list()
    dataset_url = data_url_from_file(file_path)

    suffix = "strict" if strict else "loose"

    client.datasets.register(
        dataset_def={
            "identifier": "eval-mmlu-{suffix}",
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

    # test eval with individual rows
    rows_paginated = client.datasetio.get_rows_paginated(
        dataset_id="eval-mmlu-{suffix}",
        rows_in_page=3,
        page_token=None,
        filter_condition=None,
    )

    eval_candidate = {
        "type": "model",
        "model": "Llama3.1-8B-Instruct",
        "sampling_params": {
            "strategy": "greedy",
            "temperature": 0,
            "top_p": 0.95,
            "top_k": 0,
            "max_tokens": 0,
            "repetition_penalty": 1.0,
        },
    }

    if strict:
        scoring_functions = ["meta-reference::equality"]
    else:
        scoring_functions = ["meta-reference::answer_parsing_multiple_choice"]

    eval_rows = client.eval.evaluate(
        input_rows=rows_paginated.rows,
        candidate=eval_candidate,
        scoring_functions=scoring_functions,
    )

    cprint(eval_rows, "green")


def main(
    host: str,
    port: int,
    dataset_path: Optional[str] = None,
    dataset="meta-llama/Llama-3.2-1B-Instruct-evals",
    dataset_name="Llama-3.2-1B-Instruct-evals__mmlu__details",
    strict=False,
):
    if dataset_path:
        run_eval(host, port, dataset_path, strict)
    else:
        suffix = "strict" if strict else "loose"
        dataset_path = (
            Path(os.path.abspath(__file__)).parent / f"{dataset_name}-{suffix}.csv"
        )
        process_hf_dataset(dataset, dataset_name, dataset_path, strict)
        run_eval(host, port, dataset_path, strict)


if __name__ == "__main__":
    fire.Fire(main)
