# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import base64
import mimetypes
import os

import gradio as gr
from examples.interior_design_assistant.api import InterioAgent


API = None
HOST = "localhost"
PORT = 5000
PATH = "examples/interior_design_assistant/resources/documents"
IMG_DIR = "examples/interior_design_assistant/resources/images/fireplaces"


def initialize():
    global API, HOST, PORT, PATH, IMG_DIR
    API = InterioAgent(PATH, IMG_DIR)
    asyncio.run(API.initialize(HOST, PORT))


def update_item_list(options):
    return gr.update(choices=options, value=options[0] if options else None)


def image_upload_handle(file_path):
    global API
    result = asyncio.run(API.list_items(file_path))
    return (
        result["description"],
        gr.update(visible=True),
        result["items"],
        update_item_list(result["items"]),
        gr.update(visible=True),
    )


def update_alternatives_list(options):
    return gr.update(choices=options, value=options[0] if options else None)


def suggest_alternatives(file_path, item):
    alternatives = asyncio.run(API.suggest_alternatives(file_path, item))
    return alternatives, update_alternatives_list(alternatives), gr.update(visible=True)


def lookup_button_handle(suggestion):
    res = asyncio.run(API.retrieve_images(suggestion))
    images_with_descriptions = []
    for r in res:
        path = r["image"]
        path = path.replace("<uri>", "")
        path = path.replace("</uri>", "")
        path = os.path.basename(path)
        images_with_descriptions.append(
            (
                os.path.join(IMG_DIR, path),
                r["description"],
            )
        )
    return images_with_descriptions, gr.update(visible=True)


def update_suggestion_input(selected_replacement):
    # Update the suggestion input box with the selected replacement
    return selected_replacement


with gr.Blocks() as demo:
    gr.Markdown("## Interio ")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="filepath")
            description_text_box = gr.Textbox(label="Description", visible=False)

        with gr.Column():
            with gr.Row(visible=False) as alternatives_row:
                with gr.Column():
                    object_list = gr.Radio(label="Detected Objects", choices=[])
                    replace_button = gr.Button("Suggest Alternatives")
            with gr.Row(visible=False) as suggestions_row:
                with gr.Column():
                    suggestion_list = gr.Radio(
                        label="Replacement Suggestions",
                        choices=[],
                    )
                    lookup_button = gr.Button("Look up options")

        with gr.Column():
            retrieved_images = gr.Gallery(
                label="Retrieved Images", columns=2, rows=2, visible=False
            )

    image_input.change(
        image_upload_handle,
        inputs=[image_input],
        outputs=[
            description_text_box,
            description_text_box,
            object_list,
            object_list,
            alternatives_row,
        ],
    )
    replace_button.click(
        suggest_alternatives,
        inputs=[image_input, object_list],
        outputs=[suggestion_list, suggestion_list, suggestions_row],
    )
    lookup_button.click(
        lookup_button_handle,
        inputs=[suggestion_list],
        outputs=[retrieved_images, retrieved_images],
    )

initialize()
demo.launch(server_name="0.0.0.0", server_port=7860)
