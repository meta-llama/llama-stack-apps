# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from llama_models.llama3.api.datatypes import *  # noqa: F403
from llama_stack.apis.agents import *  # noqa: F403
from llama_stack.apis.agents.client import AgentsClient
from llama_stack.apis.memory import *  # noqa: F403
from llama_stack.apis.safety import *  # noqa: F403
from dotenv import load_dotenv

from .custom_tools import CustomTool
from .execute_with_custom_tools import AgentWithCustomToolExecutor

load_dotenv()


class AttachmentBehavior(Enum):
    rag = "rag"
    code_interpreter = "code_interpreter"
    auto = "auto"


class ApiKeys(BaseModel):
    wolfram_alpha: Optional[str] = None
    brave: Optional[str] = None
    bing: Optional[str] = None


def load_api_keys_from_env() -> ApiKeys:
    return ApiKeys(
        bing=os.getenv("BING_SEARCH_API_KEY"),
        brave=os.getenv("BRAVE_SEARCH_API_KEY"),
        wolfram_alpha=os.getenv("WOLFRAM_ALPHA_API_KEY"),
    )


def search_tool_defn(api_keys: ApiKeys) -> SearchToolDefinition:
    if not api_keys.brave and not api_keys.bing:
        raise ValueError("You must specify either Brave or Bing search API key")

    return SearchToolDefinition(
        engine=SearchEngineType.bing if api_keys.bing else SearchEngineType.brave,
        api_key=api_keys.bing if api_keys.bing else api_keys.brave,
    )


def default_builtins(api_keys: ApiKeys) -> List[ToolDefinitionCommon]:
    return [
        search_tool_defn(api_keys),
        WolframAlphaToolDefinition(api_key=api_keys.wolfram_lpha),
        PhotogenToolDefinition(),
        CodeInterpreterToolDefinition(),
    ]


class QuickToolConfig(BaseModel):
    custom_tools: List[CustomTool] = Field(default_factory=list)

    prompt_format: ToolPromptFormat = ToolPromptFormat.json

    # use this to control whether you want the model to write code to
    # process them, or you want to "RAG" them beforehand
    attachment_behavior: Optional[AttachmentBehavior] = None

    builtin_tools: List[ToolDefinitionCommon] = Field(default_factory=list)

    # if you have a memory bank already pre-populated, specify it here
    memory_bank_id: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def enable_memory_tool(cfg: QuickToolConfig) -> bool:
    if cfg.memory_bank_id:
        return True
    return (
        cfg.attachment_behavior
        and cfg.attachment_behavior != AttachmentBehavior.code_interpreter
    )


# This is a utility function; it does not provide all bells and whistles
# you can get from the underlying Agents API. Any limitations should
# ideally be resolved by making another well-scoped utility function instead
# of adding complex options here.
async def make_agent_config_with_custom_tools(
    model: str = "Meta-Llama3.1-8B-Instruct",
    disable_safety: bool = False,
    tool_config: QuickToolConfig = QuickToolConfig(),
) -> AgentConfig:
    tool_definitions = []

    # ensure code interpreter is enabled if attachments need it
    builtin_tools = tool_config.builtin_tools
    tool_choice = ToolChoice.auto
    if tool_config.attachment_behavior == AttachmentBehavior.code_interpreter:
        if not any(isinstance(t, CodeInterpreterToolDefinition) for t in builtin_tools):
            builtin_tools.append(CodeInterpreterToolDefinition())

        tool_choice = ToolChoice.required

    tool_definitions = [*builtin_tools]

    if enable_memory_tool(tool_config):
        bank_configs = []
        if tool_config.memory_bank_id:
            bank_configs.append(
                AgentVectorMemoryBankConfig(bank_id=tool_config.memory_bank_id)
            )
        tool_definitions.append(MemoryToolDefinition(memory_bank_configs=bank_configs))

    tool_definitions += [t.get_tool_definition() for t in tool_config.custom_tools]

    if not disable_safety:
        for t in tool_definitions:
            t.input_shields = [ShieldDefinition(shield_type=BuiltinShield.llama_guard)]
            t.output_shields = [
                ShieldDefinition(shield_type=BuiltinShield.llama_guard),
                ShieldDefinition(shield_type=BuiltinShield.injection_shield),
            ]

    cfg = AgentConfig(
        model=model,
        instructions="You are a helpful assistant",
        sampling_params=SamplingParams(),
        tools=tool_definitions,
        tool_prompt_format=tool_config.prompt_format,
        tool_choice=tool_choice,
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
    )
    return cfg


async def get_agent_with_custom_tools(
    host: str,
    port: int,
    agent_config: AgentConfig,
    custom_tools: List[CustomTool],
) -> AgentWithCustomToolExecutor:
    api = AgentsClient(base_url=f"http://{host}:{port}")

    create_response = await api.create_agent(agent_config)
    agent_id = create_response.agent_id

    name = f"Session-{uuid.uuid4()}"
    response = await api.create_agent_session(
        agent_id=agent_id,
        session_name=name,
    )
    session_id = response.session_id

    return AgentWithCustomToolExecutor(
        api, agent_id, session_id, agent_config, custom_tools
    )
