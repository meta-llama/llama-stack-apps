# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)

from ....base_tool import BaseTool


class ModifyUserAddressTool(BaseTool):
    def get_name(self) -> str:
        return "modify_user_address"

    def get_description(self) -> str:
        return "Modify the default address of a user. The agent needs to explain the modification detail and ask for explicit user confirmation (yes/no) to proceed."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "user_id": ToolParamDefinitionParam(
                param_type="string",
                description="The user id, such as 'sara_doe_496'.",
                required=True,
            ),
            "address1": ToolParamDefinitionParam(
                param_type="string",
                description="The first line of the address, such as '123 Main St'.",
                required=True,
            ),
            "address2": ToolParamDefinitionParam(
                param_type="string",
                description="The second line of the address, such as 'Apt 1' or ''.",
                required=True,
            ),
            "city": ToolParamDefinitionParam(
                param_type="string",
                description="The city, such as 'San Francisco'.",
                required=True,
            ),
            "state": ToolParamDefinitionParam(
                param_type="string",
                description="The province, such as 'CA'.",
                required=True,
            ),
            "country": ToolParamDefinitionParam(
                param_type="string",
                description="The country, such as 'USA'.",
                required=True,
            ),
            "zip": ToolParamDefinitionParam(
                param_type="string",
                description="The zip code, such as '12345'.",
                required=True,
            ),
        }

    def run_impl(
        self,
        user_id: str,
        address1: str,
        address2: str,
        city: str,
        state: str,
        country: str,
        zip: str,
    ) -> str:
        data = self.database.data
        users = data["users"]
        if user_id not in users:
            return "Error: user not found"
        user = users[user_id]
        user["address"] = {
            "address1": address1,
            "address2": address2,
            "city": city,
            "state": state,
            "country": country,
            "zip": zip,
        }
        return json.dumps(user)