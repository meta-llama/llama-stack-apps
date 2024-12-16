# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)

from ....base_tool import BaseTool


class ThinkTool(BaseTool):
    def get_name(self) -> str:
        return "think"

    def get_description(self) -> str:
        return (
            "Use the tool to think about something. It will not obtain new information or change the database, "
            "but just append the thought to the log. Use it when complex reasoning or some cache memory is needed."
        )

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "thought": ToolParamDefinitionParam(
                param_type="string",
                description="A thought to think about.",
                required=True,
            ),
        }

    def run_impl(self, thought: str) -> str:
        # This method does not change the state of the data; it simply returns an empty string.
        return ""
