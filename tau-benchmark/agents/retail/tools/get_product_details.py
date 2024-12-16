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


class GetProductDetailsTool(BaseTool):
    def get_name(self) -> str:
        return "get_product_details"

    def get_description(self) -> str:
        return "Get the inventory details of a product."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "product_id": ToolParamDefinitionParam(
                param_type="string",
                description="The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                required=True,
            ),
        }

    def run_impl(self, product_id: str) -> str:
        data = self.database.data
        products = data["products"]
        if product_id in products:
            return json.dumps(products[product_id])
        return "Error: product not found"
