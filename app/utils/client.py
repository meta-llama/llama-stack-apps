# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import sys

THIS_DIR = os.path.dirname(__file__)
sys.path += os.path.abspath(THIS_DIR + "../../")

from common.client_utils import (
    default_builtins,
    get_agent_with_custom_tools,
    load_api_keys_from_env,
    make_agent_config_with_custom_tools,
    QuickToolConfig,
)


class ClientManager:
    _instance = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ClientManager, cls).__new__(cls)
        return cls._instance

    def init_client(
        self, inference_port, host, custom_tools=None, disable_safety=False
    ):
        if self.client is None:
            agent_config = asyncio.run(
                make_agent_config_with_custom_tools(
                    tool_config=QuickToolConfig(
                        custom_tools=custom_tools,
                        builtin_tools=default_builtins(load_api_keys_from_env()),
                        attachment_behavior="rag",
                    ),
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
            raise Exception("client is not initialized. Please initialize it first.")
        return self.client
