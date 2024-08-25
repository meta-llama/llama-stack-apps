# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio

from llama_toolchain.agentic_system.utils import (
    get_agent_with_custom_tools,
    make_agent_config_with_custom_tools,
)

global CLIENT
CLIENT = None


class ClientManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.client = None

    def init_client(
        self, inference_port, host, custom_tools=None, disable_safety=False
    ):
        if self.client is None:
            agent_config = asyncio.run(
                make_agent_config_with_custom_tools(
                    custom_tools=custom_tools,
                    disable_safety=disable_safety,
                )
            )
            self.client = asyncio.run(
                get_agent_with_custom_tools(
                    host=host,
                    port=inference_port,
                    agent_config=agent_config,
                    custom_tools=custom_tools or [],
                )
            )

    def get_client(self):
        if self.client is None:
            raise Exception("CLIENT is not initialized. Please initialize it first.")
        return self.client
