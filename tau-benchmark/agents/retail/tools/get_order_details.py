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


class GetOrderDetailsTool(BaseTool):
    def get_name(self) -> str:
        return "get_order_details"

    def get_description(self) -> str:
        return "Get the status and details of an order."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "order_id": ToolParamDefinitionParam(
                param_type="string",
                description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                required=True,
            ),
        }

    def run_impl(self, order_id: str) -> str:
        data = self.database.data
        orders = data["orders"]
        if order_id in orders:
            return json.dumps(orders[order_id])
        return "Error: order not found"
