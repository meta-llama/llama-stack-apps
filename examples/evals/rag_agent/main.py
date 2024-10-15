# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
import fire
from .agent import *
from llama_stack_client.lib.agents.event_logger import EventLogger
from termcolor import cprint
import json
import pandas

import pickle
from pathlib import Path

def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


async def test_agent(host: str, port: int, file_dir: str):
    agent = await get_rag_agent(host, port, file_dir)
    user_prompts = [
        "What are the top 5 topics that were explained in the documentation? Only list succinct bullet points.",
    ]

    for prompt in user_prompts:
        cprint(f"User> {prompt}", color="white", attrs=["bold"])
        response = agent.execute_turn(content=prompt)
        async for log in EventLogger().log(response):
            if log is not None:
                log.print()


async def bulk_generate(host: str, port: int, file_dir: str, app_dataset_path: str):
    agent = await get_rag_agent(host, port, file_dir)

    # prepare evals dataset (expected answer, generated_answer, input_query)
    evals_data = []

    # read in dataset
    df = pandas.read_excel(app_dataset_path)
    print(df.keys())

    # start generation
    for index, row in df.iterrows():
        input_query = row['query']
        print(f"Processing row {index}... {input_query}")
        expected_answer = row['expected_answer']
        # actual generation (running app)
        agent.create_new_session()
        response = await agent.execute_turn_non_streaming(content=input_query)
        generated_answer = response.content
        evals_data.append([expected_answer, generated_answer, input_query])

    # save generated dataset in format (expected_answer, generated_answer)
    output_df = pandas.DataFrame(evals_data, columns=['expected_answer', 'generated_answer', 'input_query'])
    print(output_df)
    output_df.to_excel('rag_evals.xlsx', index=False)
    print("Done")


def eval(host: str, port: int, eval_dataset_path: str):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
    )
    
    # register dataset
    response = client.datasets.create(
        dataset_def={
            "type": "custom",
            "identifier": "rag-evals",
            "url": data_url_from_file(eval_dataset_path),
        }
    )
    cprint(f"datasets/create: {response}", "cyan")

    # run evals on registered dataset
    response = client.evals.run_scorer(
        dataset_config={
            "dataset_identifier": "rag-evals",
            # "row_limit": 5,
        },
        eval_scoring_config={
            "scorer_config_list": [
                {
                    "scorer_name": "accuracy",
                },
                {
                    "scorer_name": "braintrust::answer-correctness",
                }
            ]
        }
    )
    
    for k, v in response.eval_result.metrics.items():
        cprint(f"{k}: {v}", "green")

def main(
    host: str, 
    port: int, 
    file_dir: str = "", 
    app_dataset_path: str = "", 
    eval_dataset_path: str = ""):
    # asyncio.run(test_agent(host, port, file_dir))

    if app_dataset_path != "" and file_dir != "":
        asyncio.run(bulk_generate(host, port, file_dir, app_dataset_path))
    elif eval_dataset_path != "":
        eval(host, port, eval_dataset_path)
    else:
        raise Exception("Please provide either app_dataset_path + file_dir for generation or eval_dataset_path")


if __name__ == "__main__":
    fire.Fire(main)
