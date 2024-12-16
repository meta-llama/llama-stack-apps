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

from .base import BaseRetailTool


class CancelPendingOrderTool(BaseRetailTool):
    def get_name(self) -> str:
        return "cancel_pending_order"

    def get_description(self) -> str:
        return (
            "Cancel a pending order. If the order is already processed or delivered, "
            "it cannot be cancelled. The agent needs to explain the cancellation detail "
            "and ask for explicit user confirmation (yes/no) to proceed. If the user confirms, "
            "the order status will be changed to 'cancelled' and the payment will be refunded. "
            "The refund will be added to the user's gift card balance immediately if the payment "
            "was made using a gift card, otherwise the refund would take 5-7 business days to process. "
            "The function returns the order details after the cancellation."
        )

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "order_id": ToolParamDefinitionParam(
                param_type="string",
                description="The order id, such as '#W0000000'. Be careful there is a '#' symbol at the beginning of the order id.",
                required=True,
            ),
            "reason": ToolParamDefinitionParam(
                param_type="string",
                description="The reason for cancellation, which should be either 'no longer needed' or 'ordered by mistake'.",
                required=True,
            ),
        }

    def run_impl(self, order_id: str, reason: str) -> str:
        data = self.database.data
        # check order exists and is pending
        orders = data["orders"]
        if order_id not in orders:
            return "Error: order not found"
        order = orders[order_id]
        if order["status"] != "pending":
            return "Error: non-pending order cannot be cancelled"

        # check reason
        if reason not in ["no longer needed", "ordered by mistake"]:
            return "Error: invalid reason"

        # handle refund
        refunds = []
        for payment in order["payment_history"]:
            payment_id = payment["payment_method_id"]
            refund = {
                "transaction_type": "refund",
                "amount": payment["amount"],
                "payment_method_id": payment_id,
            }
            refunds.append(refund)
            if "gift_card" in payment_id:  # refund to gift card immediately
                payment_method = data["users"][order["user_id"]]["payment_methods"][
                    payment_id
                ]
                payment_method["balance"] += payment["amount"]
                payment_method["balance"] = round(payment_method["balance"], 2)

        # update order status
        order["status"] = "cancelled"
        order["cancel_reason"] = reason
        order["payment_history"].extend(refunds)

        return json.dumps(order)
