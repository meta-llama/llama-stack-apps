# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import os
from typing import List

import pandas as pd
from llama_stack_client.lib.agents.agent import Agent
from tqdm import tqdm


async def get_response_row(agent: Agent, input_query: str) -> str:
    # single turn, each prompt is a new session
    session_id = agent.create_session(f"session-{input_query}")
    response = agent.create_turn(
        messages=[
            {
                "role": "user",
                "content": input_query,
            }
        ],
        session_id=session_id,
    )

    async for chunk in response:
        event = chunk.event
        event_type = event.payload.event_type
        if event_type == "turn_complete":
            return event.payload.turn.output_message.content


def data_url_from_file(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        file_content = file.read()

    base64_content = base64.b64encode(file_content).decode("utf-8")
    mime_type, _ = mimetypes.guess_type(file_path)

    data_url = f"data:{mime_type};base64,{base64_content}"

    return data_url


async def agent_bulk_generate(agent: Agent, input_file_path: str) -> None:
    """Reads a list of input queries and generates responses using the provided agent"""
    # load dataset and generate responses for the RAG agent
    df = pd.read_csv(input_file_path)
    user_prompts = df["input_query"].tolist()

    llamastack_generated_responses = []

    for prompt in tqdm(user_prompts):
        prompt += " Please use search tool."
        try:
            generated_response = await get_response_row(agent, prompt)
            llamastack_generated_responses.append(generated_response)
        except Exception as e:
            print(f"Error generating response for {prompt}: {e}")
            llamastack_generated_responses.append(None)
        print(
            f"Generating response for: {prompt}. Generated response: {generated_response}"
        )

    df["generated_answer"] = llamastack_generated_responses

    output_file_path = input_file_path.replace(".csv", "_llamastack_generated.csv")
    df.to_csv(output_file_path, index=False)
    print(f"Saved to {output_file_path}")
