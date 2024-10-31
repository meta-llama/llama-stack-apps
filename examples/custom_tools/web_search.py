# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from typing import Dict

from common.custom_tools import SingleMessageCustomTool
from llama_models.llama3.api.datatypes import ToolParamDefinition
from llama_stack.providers.impls.meta_reference.agents.tools.builtin import BraveSearch
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)


class WebSearchTool(SingleMessageCustomTool):
    """Tool to search web for queries"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.engine = BraveSearch(api_key)

    def get_name(self) -> str:
        return "web_search"

    def get_description(self) -> str:
        return "Search the web for a given query"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "query": ToolParamDefinitionParam(
                param_type="str",
                description="The query to search for",
                required=True,
            )
        }

    async def run_impl(self, query: str):
        return await self.engine.search(query)
