# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import json
from typing import List

from llama_stack_client.lib.agents.custom_tool import CustomTool
from llama_stack_client.types import CompletionMessage, ToolResponseMessage

from ...envs.database import RetailDatabaseEnv


class BaseRetailTool(CustomTool):
    def __init__(self, database: RetailDatabaseEnv):
        super().__init__()
        self.database = database

    def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:
        assert len(messages) == 1, "Expected single message"
        message = messages[0]
        tool_call = message.tool_calls[0]
        try:
            response = self.run_impl(**tool_call.arguments)
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
