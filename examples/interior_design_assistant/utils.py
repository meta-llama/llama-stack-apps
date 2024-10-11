# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import uuid


# TODO: This should move into a common util as will be needed by all apps
def data_url_from_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url


def create_single_turn(client, agent_config, messages):
    """Create a single turn agent session and return the response"""
    response = client.agents.create(agent_config=agent_config)
    agent_id = response.agent_id

    response = client.agents.session.create(
        agent_id=agent_id,
        session_name=uuid.uuid4().hex,
    )
    session_id = response.session_id

    generator = client.agents.turn.create(
        agent_id=agent_id,
        session_id=session_id,
        messages=messages,
        stream=True,
    )

    for chunk in generator:
        payload = chunk.event.payload
        if payload.event_type == "turn_complete":
            turn = payload.turn
    print(type(turn))
    return turn.output_message.content
