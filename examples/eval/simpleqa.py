# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
import os
from pathlib import Path
from typing import Optional

import blobfile as bf

import fire
import pandas

from datasets import load_dataset

from llama_stack_client import LlamaStackClient
from termcolor import cprint
from tqdm import tqdm

from .client import data_url_from_file


def process_dataset(dataset_path):
    df = pandas.read_csv(
        "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv"
    )
    examples = [row.to_dict() for _, row in df.iterrows()]

    # transform it into accceptable format for evals
    # <input_query, chat_completion_input, expected_output>
    input_query = []
    chat_completion_input = []
    expected_output = []

    for x in examples:
        input_query.append(x["problem"])
        chat_completion_input.append(
            [
                {
                    "role": "user",
                    "content": x["problem"],
                }
            ]
        )
        expected_output.append(x["answer"])

    transformed_data = {
        "input_query": input_query,
        "chat_completion_input": chat_completion_input,
        "expected_answer": expected_output,
    }

    df = pandas.DataFrame(transformed_data)
    dataset_path = Path(os.path.abspath(__file__)).parent / "simpleqa-input.csv"
    df.to_csv(dataset_path, index=False)
    print(f"Saved dataset at {dataset_path}!")


def run_eval(host: str, port: int, file_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )
    providers = client.providers.list()
    dataset_url = data_url_from_file(file_path)

    client.datasets.register(
        dataset_def={
            "identifier": "eval-simpleqa",
            "provider_id": providers["datasetio"][0].provider_id,
            "url": {"uri": dataset_url},
            "dataset_schema": {
                "expected_answer": {"type": "string"},
                "input_query": {"type": "string"},
                "chat_completion_input": {"type": "string"},
            },
        }
    )

    rows_paginated = client.datasetio.get_rows_paginated(
        dataset_id="eval-simpleqa",
        rows_in_page=-1,
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

    scoring_functions = ["meta-reference::llm_as_judge_openai_simpleqa"]

    output_res = {
        "chat_completion_input": [],
        "generated_output": [],
        "expected_output": [],
    }
    for x in scoring_functions:
        output_res[x] = []

    # for i in tqdm(range(len(rows_paginated.rows))):
    for i in tqdm(range(10)):
        row = rows_paginated.rows[i]
        eval_rows = client.eval.evaluate(
            input_rows=[row],
            candidate=eval_candidate,
            scoring_functions=scoring_functions,
        )
        output_res["chat_completion_input"].append(row["chat_completion_input"])
        output_res["expected_output"].append(row["expected_answer"])
        output_res["generated_output"].append(
            eval_rows.generations[0]["generated_answer"]
        )
        for scoring_fn in scoring_functions:
            output_res[scoring_fn].append(eval_rows.scores[scoring_fn].score_rows[0])

    # dump results
    save_path = Path(os.path.abspath(__file__)).parent / "eval-result.json"
    with open(save_path, "w") as f:
        json.dump(output_res, f, indent=4)

    print(f"Eval result saved at {save_path}!")


def main(
    host: str,
    port: int,
    dataset_path: Optional[str] = None,
):
    if dataset_path:
        run_eval(host, port, dataset_path)
    else:
        dataset_path = Path(os.path.abspath(__file__)).parent / "simpleqa-input.csv"
        process_dataset(dataset_path)
        run_eval(host, port, dataset_path)


if __name__ == "__main__":
    fire.Fire(main)
