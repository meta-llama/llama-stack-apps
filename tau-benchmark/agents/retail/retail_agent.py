# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client.types.agent_create_params import AgentConfig

from ...base_agent import TauAgent
from ...base_env import BaseEnv

from .configs.wiki import WIKI
from .tools.calculate import CalculateTool
from .tools.cancel_pending_order import CancelPendingOrderTool
from .tools.exchange_delivered_order_items import ExchangeDeliveredOrderItemsTool
from .tools.find_user_id_by_email import FindUserIdByEmailTool
from .tools.find_user_id_by_name_zip import FindUserIdByNameZipTool
from .tools.get_order_details import GetOrderDetailsTool
from .tools.get_product_details import GetProductDetailsTool
from .tools.list_all_product_types import ListAllProductTypesTool


def get_retail_agent():
    env = BaseEnv("retail")
    tools = [
        CalculateTool(env),
        CancelPendingOrderTool(env),
        ExchangeDeliveredOrderItemsTool(env),
        FindUserIdByEmailTool(env),
        FindUserIdByNameZipTool(env),
        GetOrderDetailsTool(env),
        GetProductDetailsTool(env),
        ListAllProductTypesTool(env),
    ]
    agent_config = AgentConfig(
        model="meta-llama/Llama-3.1-405B-Instruct-FP8",
        instructions=WIKI,
        tools=[tool.get_tool_definition() for tool in tools],
        sampling_params={
            "strategy": "greedy",
            "temperature": 1.0,
            "top_p": 0.9,
        },
        tool_choice="auto",
        tool_prompt_format="json",
        input_shields=[],
        output_shields=[],
        enable_session_persistence=False,
    )

    retail_agent = TauAgent(
        env=env,
        tools=tools,
        agent_config=agent_config,
    )
    return retail_agent
