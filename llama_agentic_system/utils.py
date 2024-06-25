# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import uuid
from typing import Any, List, Optional

from llama_models.llama3_1.api.datatypes import (
    BuiltinTool,
    InstructModel,
    Message,
    SamplingParams,
)
from llama_toolchain.inference.api.config import InferenceConfig, RemoteImplConfig
from llama_toolchain.safety.api.config import SafetyConfig
from llama_toolchain.safety.api.datatypes import BuiltinShield, ShieldDefinition

from llama_toolchain.utils import DEFAULT_DUMP_DIR, parse_config
from omegaconf import OmegaConf

from llama_agentic_system.api.config import AgenticSystemConfig, InlineImplConfig
from llama_agentic_system.api.datatypes import (
    AgenticSystemInstanceConfig,
    AgenticSystemToolDefinition,
)
from llama_agentic_system.api.endpoints import (
    AgenticSystemCreateRequest,
    AgenticSystemSessionCreateRequest,
)

from llama_agentic_system.api_instance import get_agentic_system_api_instance

from llama_agentic_system.client import execute_with_custom_tools


def get_root_directory():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while os.path.isfile(os.path.join(current_dir, "__init__.py")):
        current_dir = os.path.dirname(current_dir)

    return current_dir


def get_config_dir():
    return os.path.join(DEFAULT_DUMP_DIR, "configs", "agentic_system")


class AgenticSystemClientWrapper:

    def __init__(self, api, system_id, custom_tools):
        self.api = api
        self.system_id = system_id
        self.custom_tools = custom_tools
        self.session_id = None

    async def create_session(self, name: str = None):
        if name is None:
            name = f"Session-{uuid.uuid4()}"

        response = await self.api.create_agentic_system_session(
            AgenticSystemSessionCreateRequest(
                system_id=self.system_id,
                session_name=name,
            )
        )
        self.session_id = response.session_id
        return self.session_id

    async def run(self, messages: List[Message], stream: bool = True):
        async for chunk in execute_with_custom_tools(
            self.api,
            self.system_id,
            self.session_id,
            messages,
            self.custom_tools,
            stream=stream,
        ):
            yield chunk


async def get_agent_system_instance(
    host: str = "localhost",
    port: int = 5000,
    custom_tools: Optional[List[Any]] = None,
    disable_safety: bool = False,
) -> AgenticSystemClientWrapper:
    custom_tools = custom_tools or []

    config_dir = get_config_dir()
    config = parse_config(config_dir, "inline")

    safety_config = None
    if config.agentic_system_config.safety_config is not None:
        safety_config = SafetyConfig(
            **OmegaConf.to_container(
                config.agentic_system_config.safety_config,
                resolve=True,
            )
        )

    sampling_params = SamplingParams()
    if config.sampling_params is not None:
        sampling_params = SamplingParams(
            **OmegaConf.to_container(
                config.sampling_params,
                resolve=True,
            )
        )

    api = await get_agentic_system_api_instance(
        AgenticSystemConfig(
            # get me an inline agentic system (i.e., agentic system runs locally)
            impl_config=InlineImplConfig(
                inference_config=InferenceConfig(
                    # but make sure it points to inference which is remote
                    impl_config=RemoteImplConfig(url=f"http://{host}:{port}")
                )
            ),
            safety_config=safety_config,
        )
    )

    tool_definitions = [
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.brave_search,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.wolfram_alpha,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.photogen,
        ),
        AgenticSystemToolDefinition(
            tool_name=BuiltinTool.code_interpreter,
        ),
    ] + [t.get_tool_definition() for t in custom_tools]

    if not disable_safety:
        for t in tool_definitions:
            t.input_shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]
            t.output_shields = [
                ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ShieldDefinition(shield_type=BuiltinShield.injection_shield),
            ]

    create_request = AgenticSystemCreateRequest(
        model=InstructModel.llama3_8b_chat,
        instance_config=AgenticSystemInstanceConfig(
            instructions="You are a helpful assistant",
            available_tools=tool_definitions,
            input_shields=(
                []
                if disable_safety
                else [
                    ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                    ShieldDefinition(shield_type=BuiltinShield.jailbreak_shield),
                ]
            ),
            output_shields=(
                []
                if disable_safety
                else [
                    ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ]
            ),
            sampling_params=sampling_params,
        ),
    )
    create_response = await api.create_agentic_system(create_request)
    return AgenticSystemClientWrapper(api, create_response.system_id, custom_tools)
