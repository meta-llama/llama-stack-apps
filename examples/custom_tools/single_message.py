# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import json
from abc import abstractmethod
from typing import List

from llama_stack_client.lib.agents.client_tool import ClientTool
from llama_stack_client.types.shared.completion_message import CompletionMessage
from llama_stack_client.types.shared.tool_response_message import ToolResponseMessage


class SingleMessageCustomTool(ClientTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

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
            role="tool",
        )
        return [message]

    @abstractmethod
    def run_impl(self, *args, **kwargs):
        raise NotImplementedError()
