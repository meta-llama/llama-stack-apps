# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import os

from typing import Optional

from llama_stack_client import LlamaStackClient


class LlamaStackEvaluation:
    def __init__(self):
        self.client = LlamaStackClient(
            base_url=os.environ.get("LLAMA_STACK_ENDPOINT", "http://localhost:5000"),
            provider_data={
                "fireworks_api_key": os.environ.get("FIREWORKS_API_KEY", ""),
                "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
                "openai_api_key": os.environ.get("OPENAI_API_KEY", ""),
            },
        )

    def list_scoring_functions(self):
        """List all available scoring functions"""
        return self.client.scoring_functions.list()

    def list_models(self):
        """List all available judge models"""
        return self.client.models.list()

    def run_scoring(
        self, row, scoring_function_ids: list[str], scoring_params: Optional[dict]
    ):
        """Run scoring on a single row"""
        if not scoring_params:
            scoring_params = {fn_id: None for fn_id in scoring_function_ids}
        return self.client.scoring.score(
            input_rows=[row], scoring_functions=scoring_params
        )
