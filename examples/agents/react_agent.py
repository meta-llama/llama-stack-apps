# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import uuid

import fire

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent


@client_tool
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


def main():
    client = LlamaStackClient(
        base_url="http://localhost:8321",
    )

    model = "meta-llama/Llama-3.1-8B-Instruct"
    print(type(torchtune))
    agent = ReActAgent(
        client=client,
        model=model,
        builtin_toolgroups=["builtin::websearch"],
        client_tools=[torchtune],
    )

    session_id = agent.create_session(f"ttest-session-{uuid.uuid4().hex}")

    response = agent.create_turn(
        messages=[{"role": "user", "content": "What's the current time in new york?"}],
        session_id=session_id,
        stream=True,
    )
    for log in EventLogger().log(response):
        log.print()

    response2 = agent.create_turn(
        messages=[{"role": "user", "content": "What is torchtune?"}],
        session_id=session_id,
        stream=True,
    )
    for log in EventLogger().log(response2):
        log.print()


if __name__ == "__main__":
    fire.Fire(main)
