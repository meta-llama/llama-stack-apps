# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel

from llama_models.llama3.api.datatypes import *  # noqa: F403

from llama_toolchain.agentic_system.event_logger import EventLogger

from llama_toolchain.agentic_system.utils import (
    get_agent_with_custom_tools,
    make_agent_config_with_custom_tools,
    QuickToolConfig,
)
from llama_toolchain.agentic_system.api import *  # noqa: F403

from termcolor import cprint

load_dotenv()


class UserTurnInput(BaseModel):
    message: UserMessage
    attachments: Optional[List[Attachment]] = None


def prompt_to_turn(
    content: str, attachments: Optional[List[Attachment]] = None
) -> UserTurnInput:
    return UserTurnInput(message=UserMessage(content=content), attachments=attachments)


async def execute_turns(
    turn_inputs: List[UserTurnInput],
    host: str = "localhost",
    port: int = 5000,
    disable_safety: bool = False,
    tool_config: QuickToolConfig = QuickToolConfig(),
):
    agent_config = await make_agent_config_with_custom_tools(
        disable_safety=disable_safety,
        tool_config=tool_config,
    )
    agent = await get_agent_with_custom_tools(
        host=host,
        port=port,
        agent_config=agent_config,
        custom_tools=tool_config.custom_tools,
    )
    while len(turn_inputs) > 0:
        turn = turn_inputs.pop(0)

        iterator = agent.execute_turn(
            [turn.message],
            turn.attachments,
        )
        cprint(f"User> {turn.message.content}", color="white", attrs=["bold"])
        async for event, log in EventLogger().log(iterator):
            if log is not None:
                log.print()
