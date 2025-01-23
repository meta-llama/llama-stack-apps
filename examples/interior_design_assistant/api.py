# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import json
import textwrap
import uuid
from pathlib import Path
from typing import List

import fire

from examples.interior_design_assistant.utils import (
    create_single_turn,
    data_url_from_image,
)

from llama_stack_client import LlamaStackClient
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.tool_runtime import QueryConfigParam

from termcolor import cprint

MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class InterioAgent:
    def __init__(self, document_dir: str, image_dir: str):
        self.document_dir = document_dir
        self.image_dir = image_dir

    async def initialize(self, host: str, port: int):
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        # setup agent for inference
        self.agent_id = await self._get_agent()
        # setup memory bank for RAG
        self.bank_id = await self.build_vector_db(self.document_dir)

    async def _get_agent(self):
        agent_config = AgentConfig(
            model=MODEL,
            instructions="",
            sampling_params={"strategy": {"type": "greedy"}},
            enable_session_persistence=True,
        )
        response = self.client.agents.create(
            agent_config=agent_config,
        )
        self.agent_id = response.agent_id
        return self.agent_id

    async def list_items(self, file_path: str) -> List[str]:
        """
        Analyze the image using multimodal llm
        and return a list of items that are present in the image.
        """
        assert (
            self.agent_id is not None
        ), "Agent not initialized, call initialize() first"
        text = textwrap.dedent(
            """
            Analyze the image to provide a 4 sentence description of the architecture and furniture items present in it.
            Description should include architectural style, design, color, patterns, textures and other prominent details.
            Examples for furniture items include (but not limited to): couch, coffee table, fireplace, etc

            Return results in the following format:
            {
                "description": 4 sentence architectural description of the image,
                "items": list of furniture items present in the image
            }

            Remember to only list furniture items you see in the image. Just suggest item names without any additional text or explanations.
            For eg. "Couch" instead of "grey sectional couch"

            Please return as suggested format, Do not return any other text or explanations.
            """
        )
        resposne = self.client.agents.session.create(
            agent_id=self.agent_id,
            session_name=uuid.uuid4().hex,
        )
        data_url = data_url_from_image(file_path)

        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": {"url": {"uri": data_url}}},
                {"type": "text", "text": text},
            ],
        }

        response = self.client.agents.turn.create(
            agent_id=self.agent_id,
            session_id=resposne.session_id,
            messages=[message],
            stream=True,
        )

        result = ""
        for chunk in response:
            payload = chunk.event.payload
            if payload.event_type == "turn_complete":
                turn = payload.turn
                break

        result = turn.output_message.content
        try:
            d = json.loads(result.strip())
        except Exception:
            cprint(f"Error parsing JSON output: {result}", color="red")
            raise

        return d

    async def suggest_alternatives(
        self, file_path: str, item: str, n: int = 3
    ) -> List[str]:
        """
        Analyze the image using multimodal llm
        and return possible alternative descriptions for the provided item.
        """
        prompt = textwrap.dedent(
            """
            For the given image, your task is to carefully examine the image to provide alternative suggestions for {item}.
            The {item} should fit well with the overall aesthetic of the room.

            Carefully analyze the image, paying attention to the overall style, design, color, patterns, materials and architectural details.
            Based on your analysis, generate a 1-2 sentence detailed alternative descritions for the {item} that would complement the room's aesthetic.
            The descriptions should be detailed mentioning the style, color, material, and any other relevant details that would help in a search query
            Each alternative should be different from each other but still fit harmoniously within the space.

            Each description should be 10-20 words long and should be on a separate line.
            Return results in the following format:
            [
                {{
                    "description": first alternative suggestion of the item
                }},
                {{
                    "description": second alternative suggestion of the item
                }},
            ]

            Only provide {n} alternative descriptions, nothing else.
            Return JSON as suggested, Do not return any other text or explanations. Don't forget the ',' at the end of each description.
            """
        )

        text = prompt.format(item=item, n=n)
        data_url = data_url_from_image(file_path)

        message = {
            "role": "user",
            "content": [
                {"type": "image", "image": {"url": {"uri": data_url}}},
                {
                    "type": "text",
                    "text": text,
                },
            ],
        }

        resposne = self.client.agents.session.create(
            agent_id=self.agent_id,
            session_name=uuid.uuid4().hex,
        )
        generator = self.client.agents.turn.create(
            agent_id=self.agent_id,
            session_id=resposne.session_id,
            messages=[message],
            stream=True,
        )
        result = ""
        for chunk in generator:
            payload = chunk.event.payload
            if payload.event_type == "turn_complete":
                turn = payload.turn

        result = turn.output_message.content
        print(result)
        return [r["description"].strip() for r in json.loads(result.strip())]

    async def retrieve_images(self, description: str):
        """
        Retrieve images from the memory bank that match the description
        """
        assert (
            self.bank_id is not None
        ), "Setup bank before calling this method via initialize()"

        agent_config = AgentConfig(
            enable_session_persistence=False,
            model=MODEL,
            instructions="",
            sampling_params={"strategy": {"type": "greedy"}},
            toolgroups=[
                # Enable memory as a tool for RAG
                {
                    "name": "builtin::memory",
                    "args": {
                        "vector_db_ids": [self.bank_id],
                        "query_config": QueryConfigParam(
                            max_chunks=5,
                            max_tokens_in_context=4096,
                            query_generator_config={
                                "type": "llm",
                                "model": MODEL,
                                "template": textwrap.dedent(
                                    """
                                You are given a conversation between a user and their assistant.
                                From this conversation, you need to extract a one sentence description that is being asked for by the user.
                                This one sentence description will be used to query a memory bank to retrieve relevant images.

                                Analyze the provided conversation and respond with one line description and no other text or explanation.

                                Here is the conversation:
                                {% for message in messages %}
                                {{ message.role }}> {{ message.content }}
                                {% endfor %}
                                """
                                ),
                            },
                        ),
                    },
                },
            ],
        )

        prompt = textwrap.dedent(
            """
            You are given a description of an item.
            Your task is to find images of that item in the memory bank that match the description.
            Return the top 4 most relevant results.

            Return results in the following format:
            [
                {
                    "image": "uri value",
                    "description": "description of the image",
                },
                {
                    "image": "uri value",
                    "description": "description of the image 2",
                }
            ]
            The uri value is enclosed in the tags <uri> and </uri>.
            The description is a summarized explanation of why this item is relevant and how it can enhance the room.

            Return JSON as suggested, Do not return any other text or explanations.
            Do not create uri values, return actual uri value (eg. "011.webp") as is.
            """
        )
        description = f"Description: {description}"
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": description},
            ],
        }

        response = create_single_turn(self.client, agent_config, [message])
        return json.loads(response.strip())

    # NOTE: If using a persistent memory bank, building on the fly is not needed
    # and LlamaStack apis can leverage existing banks
    async def build_vector_db(self, local_dir: str) -> str:
        """
        Build a vector db that can be used to store and retrieve images.
        """
        self.live_bank = "interio_bank"
        self.client.vector_dbs.register(
            vector_db_id=self.live_bank,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )

        local_dir = Path(local_dir)
        # read all files in the provided local_dir
        # amd add each file as a document in the memory bank
        documents = []
        for i, file in enumerate(local_dir.iterdir()):
            if file.is_file():
                with file.open("r") as f:
                    documents.append(
                        {
                            "document_id": uuid.uuid4().hex,
                            "content": f.read(),
                            "mime_type": "text/plain",
                        }
                    )
        # insert the documents into the memory bank
        assert len(documents) > 0, "No documents found in the provided directory"
        self.client.tool_runtime.rag_tool.insert(
            vector_db_id=self.live_bank,
            documents=documents,
            chunk_size_in_tokens=512,
        )

        return "interio_bank"


async def async_main(host: str, port: int, memory_path: str, image_dir: str):
    interio = InterioAgent(memory_path, image_dir)
    await interio.initialize(host, port)

    # Test query to ensure memory bank is working
    # query = (
    #     "A rustic, stone-faced fireplace with a wooden mantel and a cast-iron insert."
    # )
    # res = interio.client.tool_runtime.rag_tool.query(
    #     vector_db_id=interio.bank_id,
    #     query=query,
    # )
    # print(res)

    path = input(
        "Enter Image path (relative to image_dir or memory_path is accepted) >> "
    )
    path = Path(path)

    options = [
        path,
        Path(image_dir) / path,
        Path(memory_path) / path,
    ]
    chosen_path = None
    for p in options:
        if p.exists():
            chosen_path = p
            break

    if not chosen_path:
        cprint(f"No valid path found in {options}", color="red")
        return

    result = await interio.list_items(chosen_path)

    cprint(f"Here is the description: {result['description']}", color="yellow")
    cprint(f"Here are the identified items: {result['items']}", color="yellow")
    item = input("Which item do you want to change? >> ")
    alternatives = await interio.suggest_alternatives(chosen_path, item)
    alt_str = "\n- ".join(alternatives)
    cprint(f"Here are the suggested alternatives: \n- {alt_str}", color="yellow")

    choice = input("Which alternative did you like? >> ")
    res = await interio.retrieve_images(alternatives[int(choice)])

    print("Here are some ideas")
    for r in res:
        cprint(f"{r['image']}", color="green")
        cprint(f"{r['description']}", color="yellow")


def main(host: str, port: int, memory_path: str, image_dir: str):
    asyncio.run(async_main(host, port, memory_path, image_dir))


if __name__ == "__main__":
    fire.Fire(main)
