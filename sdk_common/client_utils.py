# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import os
import uuid
from enum import Enum
from typing import List, Optional

# from .custom_tools import CustomTool
# from .execute_with_custom_tools import AgentWithCustomToolExecutor

from llama_stack.types.agent_create_params import (
    AgentConfig,
    AgentConfigToolCodeInterpreterToolDefinition,
    AgentConfigToolFunctionCallToolDefinition,
    AgentConfigToolSearchToolDefinition,
    AgentConfigToolWolframAlphaToolDefinition,
)

# from llama_models.llama3.api.datatypes import *  # noqa: F403
# from llama_stack.apis.agents import *  # noqa: F403
# from llama_stack.apis.agents.client import AgentsClient
# from llama_stack.apis.memory import *  # noqa: F403
# from llama_stack.apis.safety import *  # noqa: F403

from pydantic import BaseModel, Field


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


def search_tool_defn(api_keys: ApiKeys) -> AgentConfigToolSearchToolDefinition:
    if not api_keys.brave and not api_keys.bing:
        raise ValueError("You must specify either Brave or Bing search API key")

    return AgentConfigToolSearchToolDefinition(
        type="bing" if api_keys.bing else "brave_search",
        engine="bing" if api_keys.bing else "brave",
        api_key=api_keys.bing if api_keys.bing else api_keys.brave,
    )
