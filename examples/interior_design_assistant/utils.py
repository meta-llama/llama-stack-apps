# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import mimetypes
import uuid

import json

def enforce_response_format(output, n):
    """
    Validate the JSON output from the model.

    Args:
        output (str): Raw string output from the model.
        n (int): Number of descriptions expected in the output.

    Returns:
        list: List of validated alternative descriptions.
    """
    try:
        parsed_output = json.loads(output.strip())
        # Ensure it is a list and contains valid "description" fields
        if isinstance(parsed_output, list):
            validated_descriptions = [
                item.get("description", f"Alternative suggestion {i+1}")
                for i, item in enumerate(parsed_output)
            ]
            # If there are fewer descriptions than expected, add fallback suggestions
            while len(validated_descriptions) < n:
                validated_descriptions.append(f"Alternative suggestion {len(validated_descriptions)+1}")
            return validated_descriptions
        else:
            raise ValueError("Output is not a valid list")
    except (json.JSONDecodeError, ValueError):
        # Return fallback suggestions for invalid JSON or unexpected structure
        return [f"Alternative suggestion {i+1}" for i in range(n)]

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
