# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from .calculate import CalculateTool
from .cancel_pending_order import CancelPendingOrderTool
from .exchange_delivered_order_items import ExchangeDeliveredOrderItemsTool
from .find_user_id_by_email import FindUserIdByEmailTool
from .find_user_id_by_name_zip import FindUserIdByNameZipTool
from .get_order_details import GetOrderDetailsTool
from .get_product_details import GetProductDetailsTool
from .list_all_product_types import ListAllProductTypesTool
from .modify_pending_order_address import ModifyPendingOrderAddressTool
from .modify_pending_order_items import ModifyPendingOrderItemsTool
from .modify_pending_order_payment import ModifyPendingOrderPaymentTool
from .modify_user_address import ModifyUserAddressTool


ALL_TOOLS = [
    CalculateTool,
    CancelPendingOrderTool,
    ExchangeDeliveredOrderItemsTool,
    FindUserIdByEmailTool,
    FindUserIdByNameZipTool,
    GetOrderDetailsTool,
    GetProductDetailsTool,
    ListAllProductTypesTool,
    ModifyPendingOrderAddressTool,
    ModifyPendingOrderItemsTool,
    ModifyPendingOrderPaymentTool,
    ModifyUserAddressTool,
]
