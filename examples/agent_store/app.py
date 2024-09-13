import asyncio
import os

import gradio as gr

from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.memory.api import *  # noqa: F403

from common.client_utils import *  # noqa: F403
from termcolor import cprint

from examples.agent_store.api import AgentChoice, AgentStore


MODEL = "Meta-Llama3.1-8B-Instruct"
CHATBOT = None
SELECTED_AGENT = None
BANK_ID = "5f126596-87d8-4b9f-a44d-3a5b93bfc171"


def initialize():
    global CHATBOT

    CHATBOT = AgentStore("localhost", 5000, MODEL)
    asyncio.run(CHATBOT.initialize_agents([BANK_ID]))


def respond(message, attachments, chat_history):
    global SELECTED_AGENT
    print(f"Attachements: {attachments}")
    response, inserted_context = asyncio.run(
        CHATBOT.chat(SELECTED_AGENT, message, attachments)
    )
    cprint(f"Response: {response}", "green")
    chat_history.append((message, response))
    return "", chat_history, None, inserted_context


def agent_selection(agent_choice):
    global SELECTED_AGENT
    SELECTED_AGENT = AgentChoice[agent_choice]
    print(f"Selected Agent: {SELECTED_AGENT}")
    return "", [], None, None


with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    #file-upload { height: 50px; }
    """,
) as demo:
    with gr.Row():
        choices = [
            AgentChoice.SimpleAgent.value,
            AgentChoice.AgentWithMemory.value,
            AgentChoice.AgentWithSearchAndBrowse.value,
        ]
        dropdown = gr.Dropdown(
            choices=choices,
            label="Select an Agent",
            value=choices[0],
            interactive=True,
        )
    with gr.Row():
        chatbot = gr.Chatbot(scale=3)
        data = gr.Textbox(label="Retrieved Context", scale=2)
    with gr.Row():
        prompt = gr.Textbox(placeholder="Ask a question", container=False, scale=4)
        file_input = gr.File(
            elem_id="file-upload",
            label="📎",
            file_count="multiple",
        )
        submit_button = gr.Button("Submit")

    # initialize the dropdown
    agent_selection(dropdown.value)

    prompt.submit(
        respond,
        [prompt, file_input, chatbot],
        [prompt, chatbot, file_input, data],
    )
    submit_button.click(
        respond,
        [prompt, file_input, chatbot],
        [prompt, chatbot, file_input, data],
    )
    dropdown.change(
        fn=agent_selection,
        inputs=dropdown,
        outputs=[prompt, chatbot, file_input, data],
    )

initialize()
demo.launch(server_name="0.0.0.0", server_port=7860)
