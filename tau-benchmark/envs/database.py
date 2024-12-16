# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .data import load_data


class RetailDatabaseEnv:
    def __init__(self):
        self.data = load_data()

    def get_order(self, order_id: str) -> dict:
        pass

    def update_order(self, order_id: str, order: dict) -> None:
        pass

    def get_product(self, product_id: str) -> dict:
        pass

    def update_product(self, product_id: str, product: dict) -> None:
        pass

    def cancel_pending_order(self, order_id: str, reason: str) -> None:
        self.data["orders"][order_id]["status"] = "cancelled"
