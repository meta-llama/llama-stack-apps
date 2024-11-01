# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from abc import abstractmethod
from typing import Dict, List

import yfinance as yf
from llama_stack_client.lib.agents.custom_tool import CustomTool
from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)


class SingleMessageCustomTool(CustomTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]

        try:
            response = await self.run_impl(**tool_call.arguments)
            response_str = json.dumps(response, ensure_ascii=False)
        except Exception as e:
            response_str = f"Error when running tool: {e}"

        message = ToolResponseMessage(
            call_id=tool_call.call_id,
            tool_name=tool_call.tool_name,
            content=response_str,
            role="ipython",
        )
        return [message]

    @abstractmethod
    async def run_impl(self, *args, **kwargs):
        raise NotImplementedError()


class TickerDataTool(SingleMessageCustomTool):
    """Tool to get finance data using yfinance apis"""

    def get_name(self) -> str:
        return "get_ticker_data"

    def get_description(self) -> str:
        return "Get yearly closing prices for a given ticker symbol"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "ticker_symbol": ToolParamDefinitionParam(
                param_type="str",
                description="The ticker symbol for which to get the data. eg. '^GSPC'",
                required=True,
            ),
            "start": ToolParamDefinitionParam(
                param_type="str",
                description="Start date, eg. '2021-01-01'",
                required=True,
            ),
            "end": ToolParamDefinitionParam(
                param_type="str",
                description="End date, eg. '2024-12-31'",
                required=True,
            ),
        }

    async def run_impl(self, ticker_symbol: str, start: str, end: str):
        data = yf.download(ticker_symbol, start=start, end=end)

        data["Year"] = data.index.year
        annual_close = data.groupby("Year")["Close"].last().reset_index()
        annual_close_json = annual_close.to_json(orient="records", date_format="iso")

        return annual_close_json
