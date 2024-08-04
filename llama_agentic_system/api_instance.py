# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_toolchain.inference.client import InferenceClient
from llama_toolchain.safety.client import SafetyClient

from .agentic_system import AgenticSystemImpl
from .api.endpoints import AgenticSystem
from .config import AgenticSystemConfig


async def get_agentic_system_api_instance(config: AgenticSystemConfig) -> AgenticSystem:
    distro_url = config.llama_distribution_url

    inference_api = InferenceClient(distro_url)
    safety_api = SafetyClient(distro_url)

    return AgenticSystemImpl(inference_api, safety_api)

    # soon-to-arrive only the below will be valid
    #
    # from .client import AgenticSystemClient
    # return AgenticSystemClient(config.impl_config.url)
