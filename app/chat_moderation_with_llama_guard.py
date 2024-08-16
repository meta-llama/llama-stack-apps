# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import mesop as me
from utils.chat import chat, State
from utils.client import ClientManager

from utils.common import DISABLE_SAFETY, DISTRIBUTION_HOST, DISTRIBUTION_PORT, on_attach

from utils.transform import transform


client_manager = ClientManager()
client_manager.init_client(
    inference_port=DISTRIBUTION_PORT,
    host=DISTRIBUTION_HOST,
    custom_tools=[],
    disable_safety=DISABLE_SAFETY,
)


@me.page(
    path="/",
    title="Llama Agentic System - Llama Guard Chat Moderation",
)
def page():
    state = me.state(State)
    chat(
        transform,
        title="Llama Agentic System - Llama Guard Chat Moderation",
        bot_user="Llama Agent",
        on_attach=on_attach,
        moderated=True,
    )


if __name__ == "__main__":
    import subprocess

    subprocess.run(["mesop", __file__])
