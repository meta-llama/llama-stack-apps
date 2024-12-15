# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
# import os

import fire

from envs.wiki import WIKI
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.agent_create_params import AgentConfig


def run_retail_agent():
    client = LlamaStackClient(
        base_url="http://localhost:5000",
    )

    available_models = [model.identifier for model in client.models.list()]
    model_id = "meta-llama/Llama-3.1-405B-Instruct-FP8"

    agent_config = AgentConfig(
        model=model_id,
        instructions=WIKI,
        tools=[],
    )


def main():
    run_retail_agent()


if __name__ == "__main__":
    fire.Fire(main)
