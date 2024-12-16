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


class ReturnDeliveredOrderItemsTool(BaseTool):
    def get_name(self) -> str:
        return "return_delivered_order_items"

    def get_description(self) -> str:
        return (
            "Return some items of a delivered order. The order status will be changed to 'return requested'. "
            "The agent needs to explain the return detail and ask for explicit user confirmation (yes/no) to proceed. "
            "The user will receive follow-up email for how and where to return the item."
        )

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "order_id": ToolParamDefinitionParam(
                param_type="string",
                description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                required=True,
            ),
            "item_ids": ToolParamDefinitionParam(
                param_type="array",
                description="The item ids to be returned, each such as '1008292230'. There could be duplicate items in the list.",
                required=True,
            ),
            "payment_method_id": ToolParamDefinitionParam(
                param_type="string",
                description=(
                    "The payment method id to pay or receive refund for the item price difference, such as 'gift_card_0000000' or 'credit_card_0000000'. "
                    "These can be looked up from the user or order details."
                ),
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
