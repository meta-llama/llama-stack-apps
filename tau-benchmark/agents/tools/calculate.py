# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)

from .base import BaseRetailTool


class CalculateTool(BaseRetailTool):
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

    def run_impl(self, expression: str) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            return "Error: invalid characters in expression"
        try:
            # Evaluate the mathematical expression safely
            return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))
        except Exception as e:
            return f"Error: {e}"
