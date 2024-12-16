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


class TransferToHumanAgentsTool(BaseTool):
    def get_name(self) -> str:
        return "transfer_to_human_agents"

    def get_description(self) -> str:
        return (
            "Transfer the user to a human agent, with a summary of the user's issue. "
            "Only transfer if the user explicitly asks for a human agent, or if the user's issue cannot be resolved by the agent with the available tools."
        )

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "summary": ToolParamDefinitionParam(
                param_type="string",
                description="A summary of the user's issue.",
                required=True,
            ),
        }

    def run_impl(self, summary: str) -> str:
        # This method simulates the transfer to a human agent.
        return "Transfer successful"
