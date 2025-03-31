# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import uuid

import fire
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.agents.react.tool_parser import ReActOutput
from termcolor import colored


def torchtune(query: str = "torchtune"):
    """
    Answer information about torchtune.

    :param query: The query to use for querying the internet
    :returns: Information about torchtune
    """
    dummy_response = """
    torchtune is a PyTorch library for easily authoring, finetuning and experimenting with LLMs.

    torchtune provides:

    PyTorch implementations of popular LLMs from Llama, Gemma, Mistral, Phi, and Qwen model families
    Hackable training recipes for full finetuning, LoRA, QLoRA, DPO, PPO, QAT, knowledge distillation, and more
    Out-of-the-box memory efficiency, performance improvements, and scaling with the latest PyTorch APIs
    YAML configs for easily configuring training, evaluation, quantization or inference recipes
    Built-in support for many popular dataset formats and prompt templates
    """
    return dummy_response


def get_model(client: LlamaStackClient):
    available_models = [
        model.identifier
        for model in client.models.list()
        if model.model_type == "llm" and "guard" not in model.identifier
    ]
    return available_models[0]


def main(host: str, port: int, model: str | None = None):
    client = LlamaStackClient(
        base_url=f"http://{host}:{port}",
        provider_data={"tavily_search_api_key": os.getenv("TAVILY_SEARCH_API_KEY")},
    )

    if model is None:
        model = get_model(client)

    print(colored(f"Using model: {model}", "green"))
    agent = ReActAgent(
        client=client,
        model=model,
        tools=["builtin::websearch", torchtune],
        response_format={
            "type": "json_schema",
            "json_schema": ReActOutput.model_json_schema(),
        },
    )

    session_id = agent.create_session(f"test-session-{uuid.uuid4().hex}")
    user_prompt = "Whats the best place in new york for a pizza slice at 2am ?"
    print(colored(f"User> {user_prompt}", "blue"))
    response = agent.create_turn(
        messages=[{"role": "user", "content": user_prompt}],
        session_id=session_id,
        stream=True,
    )
    for log in EventLogger().log(response):
        log.print()

    user_prompt2 = "What are the popular llms supported in torchtune?"
    print(colored(f"User> {user_prompt2}", "blue"))
    response2 = agent.create_turn(
        messages=[{"role": "user", "content": user_prompt2}],
        session_id=session_id,
        stream=True,
    )
    for log in EventLogger().log(response2):
        log.print()


if __name__ == "__main__":
    fire.Fire(main)
