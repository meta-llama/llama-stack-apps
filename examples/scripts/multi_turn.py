# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from typing import List, Optional

from dotenv import load_dotenv

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_toolchain.agentic_system.event_logger import EventLogger

from llama_toolchain.agentic_system.meta_reference.execute_with_custom_tools import (
    execute_with_custom_tools,
)
from llama_toolchain.agentic_system.utils import get_agent_system_instance
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.tools.custom.datatypes import CustomTool

from termcolor import cprint

load_dotenv()


def prompt_to_message(content: str) -> Message:
    return UserMessage(content=content)


async def run_main(
    user_messages: List[Message],
    host: str = "localhost",
    port: int = 5000,
    disable_safety: bool = False,
    custom_tools: Optional[List[CustomTool]] = None,
):
    custom_tools = custom_tools or []
    client = await get_agent_system_instance(
        host=host,
        port=port,
        disable_safety=disable_safety,
        custom_tools=custom_tools,
    )
    await client.create_session(__file__)
    while len(user_messages) > 0:
        message = user_messages.pop(0)
        iterator = execute_with_custom_tools(
            client.api,
            client.agent_id,
            client.session_id,
            [message],
            custom_tools,
        )
        cprint(f"User> {message.content}", color="blue")
        async for event, log in EventLogger().log(iterator):
            if log is not None:
                log.print()
