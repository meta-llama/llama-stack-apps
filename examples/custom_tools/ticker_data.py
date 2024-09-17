# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os
import sys

THIS_DIR = os.path.dirname(__file__)
sys.path += os.path.abspath(THIS_DIR + "../../")

from typing import Dict

import yfinance as yf

from common.custom_tools import SingleMessageCustomTool
from llama_models.llama3.api.datatypes import ToolParamDefinition


class TickerDataTool(SingleMessageCustomTool):
    """Tool to get finance data using yfinance apis"""

    def get_name(self) -> str:
        return "get_ticker_data"

    def get_description(self) -> str:
        return "Get yearly closing prices for a given ticker symbol"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        return {
            "ticker_symbol": ToolParamDefinition(
                param_type="str",
                description="The ticker symbol for which to get the data. eg. '^GSPC'",
                required=True,
            ),
            "start": ToolParamDefinition(
                param_type="str",
                description="Start date, eg. '2021-01-01'",
                required=True,
            ),
            "end": ToolParamDefinition(
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
