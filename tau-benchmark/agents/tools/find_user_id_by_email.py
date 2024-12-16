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


class FindUserIdByEmailTool(BaseRetailTool):
    def get_name(self) -> str:
        return "find_user_id_by_email"

    def get_description(self) -> str:
        return "Find user id by email. If the user is not found, the function will return an error message."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "email": ToolParamDefinitionParam(
                param_type="string",
                description="The email of the user, such as 'something@example.com'.",
                required=True,
            ),
        }

    def run_impl(self, email: str) -> str:
        users = self.database.data["users"]
        for user_id, profile in users.items():
            if profile["email"].lower() == email.lower():
                return user_id
        return "Error: user not found"
