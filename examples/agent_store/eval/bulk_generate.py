# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import asyncio
from datetime import datetime

import fire

import pandas as pd
from tqdm import tqdm

from ..api import AgentChoice, AgentStore
from ..app import MODEL


async def app_initialize(host: str, port: int, model: str, docs_dir: str):
    agent_store_app = AgentStore(host, port, model)
    bank_id = await agent_store_app.build_index(docs_dir)
    await agent_store_app.initialize_agents([bank_id])
    return agent_store_app


async def app_bulk_generate(
    host: str,
    port: int,
    model: str,
    docs_dir: str,
    dataset_path: str,
):
    agent_store_app = await app_initialize(host, port, model, docs_dir)

    # read and loop over user prompts
    df = pd.read_csv(dataset_path)
    user_prompts = df["input_query"].tolist()
    generated_responses = []
    for user_prompt in tqdm(user_prompts):
        agent_store_app.create_session(AgentChoice.Memory)
        output_msg, inserted_context = await agent_store_app.chat(
            AgentChoice.Memory, user_prompt, []
        )
        generated_responses.append(output_msg)

    # save to new csv
    new_dataset_path = dataset_path.replace(
        ".csv", f"_generated_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    )
    df["generated_answer"] = generated_responses
    df.to_csv(new_dataset_path, index=False)

    print(
        f"You may now run `llama-stack-client eval run_scoring <scoring_fn_ids> --dataset_path {new_dataset_path}` to score the generated responses."
    )


def main(
    host: str = "localhost",
    port: int = 5000,
    model: str = MODEL,
    docs_dir: str = "",
    dataset_path: str = "",
):
    asyncio.run(app_bulk_generate(host, port, model, docs_dir, dataset_path))


if __name__ == "__main__":
    fire.Fire(main)
