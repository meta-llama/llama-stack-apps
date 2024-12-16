# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright Sierra

import json
import os
from typing import Any

FOLDER_PATH = os.path.dirname(__file__)


def load_data() -> dict[str, Any]:
    with open(os.path.join(FOLDER_PATH, "orders.json")) as f:
        order_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "products.json")) as f:
        product_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "users.json")) as f:
        user_data = json.load(f)
    return {
        "orders": order_data,
        "products": product_data,
        "users": user_data,
    }
