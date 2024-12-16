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


class ListAllProductTypesTool(BaseTool):
    def get_name(self) -> str:
        return "list_all_product_types"

    def get_description(self) -> str:
        return "List the name and product id of all product types. Each product type has a variety of different items with unique item ids and options. There are only 50 product types in the store."

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "product_id": ToolParamDefinitionParam(
                param_type="string",
                description="The product id, such as '6086499569'. Be careful the product id is different from the item id.",
                required=True,
            ),
        }

    def run_impl(self) -> str:
        data = self.database.data
        products = data["products"]
        product_dict = {
            product["name"]: product["product_id"] for product in products.values()
        }
        product_dict = dict(sorted(product_dict.items()))
        return json.dumps(product_dict)
