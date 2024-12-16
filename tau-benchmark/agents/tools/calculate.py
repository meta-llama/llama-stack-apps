# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import Dict, List

from llama_stack_client.lib.agents.custom_tool import CustomTool
from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)

from ...envs.database import RetailDatabaseEnv


class CalculateTool(CustomTool):
    def __init__(self, database: RetailDatabaseEnv):
        super().__init__()
        self.database = database

    def get_name(self) -> str:
        return "calculate"

    def get_description(self) -> str:
        return "Calculate the result of a mathematical expression."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "expression": ToolParamDefinitionParam(
                param_type="string",
                description="The mathematical expression to calculate, such as '2 + 2'. The expression can contain numbers, operators (+, -, *, /), parentheses, and spaces.",
                required=True,
            ),
        }

    def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"
        message = messages[0]
        tool_call = message.tool_calls[0]
        try:
            response = self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"
        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
            role="ipython",
        )
        return [message]

    def run_impl(self, expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: invalid characters in expression"
        try:
            # Evaluate the mathematical expression safely
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error: {e}"
