import asyncio
import base64
import mimetypes
import os

import fire

from llama_stack_client import LlamaStackClient
from termcolor import cprint


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


async def run_main(host: str, port: int, file_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )

    providers = client.providers.list()
    dataset_url = data_url_from_file(file_path)

    client.datasets.register(
        dataset_def={
            "identifier": "eval-dataset",
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
        dataset_id="eval-dataset",
        rows_in_page=3,
        page_token=None,
        filter_condition=None,
    )
    print(rows_paginated)

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
            "meta-reference::llm_as_judge_8b_correctness",
        ],
    )
    cprint(eval_rows, "green")

    # # check scoring functions available
    # score_fn_list = client.scoring_functions.list()
    # cprint([x.identifier for x in score_fn_list], "green")

    # score_rows = client.scoring.score(
    #     input_rows=rows_paginated.rows,
    #     scoring_functions=["meta-reference::equality"],
    # )
    # cprint(f"Score Rows: {score_rows}", "red")

    # # test scoring batch with dataset id
    # score_batch = client.scoring.score_batch(
    #     dataset_id="test-dataset",
    #     scoring_functions=[x.identifier for x in score_fn_list],
    #     save_results_dataset=False,
    # )

    # cprint(f"Score Batch: {score_batch}", "yellow")


def main(host: str, port: int, file_path: str):
    asyncio.run(run_main(host, port, file_path))


if __name__ == "__main__":
    fire.Fire(main)
