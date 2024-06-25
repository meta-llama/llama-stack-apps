# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .api.config import AgenticSystemConfig, ImplType
from .api.endpoints import AgenticSystem


async def get_agentic_system_api_instance(config: AgenticSystemConfig) -> AgenticSystem:
    if config.impl_config.impl_type == ImplType.inline.value:
        from llama_toolchain.inference.api_instance import get_inference_api_instance

        from .agentic_system import AgenticSystemImpl

        inference_api = await get_inference_api_instance(
            config.impl_config.inference_config
        )
        return AgenticSystemImpl(config.safety_config, inference_api)

    from .client import AgenticSystemClient

    return AgenticSystemClient(config.impl_config.url)
