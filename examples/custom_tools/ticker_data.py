# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
from typing import Dict

import yfinance as yf
from llama_stack_client.types.tool_def_param import Parameter

from .single_message import SingleMessageCustomTool


class TickerDataTool(SingleMessageCustomTool):
    """Tool to get finance data using yfinance apis"""

    def get_name(self) -> str:
        return "get_ticker_data"

    def get_description(self) -> str:
        return "Get yearly closing prices for a given ticker symbol"

    def get_params_definition(self) -> Dict[str, Parameter]:
        return {
            "ticker_symbol": Parameter(
                name="ticker_symbol",
                parameter_type="str",
                description="The ticker symbol for which to get the data. eg. '^GSPC'",
                required=True,
            ),
            "start": Parameter(
                name="start",
                parameter_type="str",
                description="Start date, eg. '2021-01-01'",
                required=True,
            ),
            "end": Parameter(
                name="end",
                parameter_type="str",
                description="End date, eg. '2024-12-31'",
                required=True,
            ),
        }

    def run_impl(self, ticker_symbol: str, start: str, end: str):
        data = yf.download(ticker_symbol, start=start, end=end)

        data["Year"] = data.index.year
        annual_close = data.groupby("Year")["Close"].last().reset_index()
        annual_close_json = annual_close.to_json(orient="records", date_format="iso")

        return annual_close_json
