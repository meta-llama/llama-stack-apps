import asyncio
import os

import fire

import gradio as gr

# from llama_toolchain.agentic_system.api import *  # noqa: F403
# from llama_toolchain.memory.api import *  # noqa: F403

from common.client_utils import *  # noqa: F403
from termcolor import cprint

from examples.agent_store.api import AgentChoice, AgentStore


MODEL = "Meta-Llama3.1-8B-Instruct"
CHATBOT = None
SELECTED_AGENT = None
BANK_ID = "5f126596-87d8-4b9f-a44d-3a5b93bfc171"
CHAT_HISTORY = {}
CONTEXT = {}


def initialize(host: str, port: int, model: str, bank_id_str: str):
    global CHATBOT

    CHATBOT = AgentStore(host, port, model)
    if bank_id_str:
        bank_ids = bank_id_str.split(",")
    else:
        bank_ids = []
    asyncio.run(CHATBOT.initialize_agents(bank_ids))


def respond(message, attachments, chat_history):
    global SELECTED_AGENT, CONTEXT, CHAT_HISTORY
    response, inserted_context = asyncio.run(
        CHATBOT.chat(SELECTED_AGENT, message, attachments)
    )
    chat_history.append((message, response))
    CHAT_HISTORY[SELECTED_AGENT] = chat_history
    CONTEXT[SELECTED_AGENT] = inserted_context
    return "", chat_history, None, inserted_context


def agent_selection(agent_choice):
    global SELECTED_AGENT, CONTEXT, CHAT_HISTORY
    SELECTED_AGENT = AgentChoice[agent_choice]
    # print(f"Selected Agent: {SELECTED_AGENT}")
    return (
        "",
        CHAT_HISTORY.get(SELECTED_AGENT, ""),
        None,
        CONTEXT.get(SELECTED_AGENT, ""),
    )


def clear_chat_button_handler():
    global CHAT_HISTORY, CONTEXT, SELECTED_AGENT
    CHAT_HISTORY[SELECTED_AGENT] = []
    CONTEXT[SELECTED_AGENT] = ""
    # create new sessions for agents
    CHATBOT.create_session(SELECTED_AGENT)
    return [], "", None


def like_button_handler(chat_history, context):
    text = ""
    for q, a in chat_history:
        text += f"User> {q}\n"
        text += f"AI> {a}\n"

    text += f"Additional Context: {context} \n"
    CHATBOT.append_to_live_memory_bank(text)


def clear_bank_button_handler():
    asyncio.run(CHATBOT.clear_live_bank())
    return [], "", None


with gr.Blocks(
    theme=gr.themes.Soft(),
    css="""
    #file-upload { height: 50px; }
    """,
) as demo:
    with gr.Row():
        choices = [
            AgentChoice.WebSearch.value,
            AgentChoice.Memory.value,
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
    with gr.Row():
        like_butoon = gr.Button("Ingest into Memory Bank")
        clear_chat = gr.Button("Clear Chat")
        clear_bank = gr.Button("Clear Bank")

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
    like_butoon.click(like_button_handler, inputs=[chatbot, data])
    clear_chat.click(
        clear_chat_button_handler, inputs=[], outputs=[chatbot, data, file_input]
    )
    clear_bank.click(
        clear_bank_button_handler, inputs=[], outputs=[chatbot, data, file_input]
    )


def main(
    host: str = "localhost",
    port: int = 5000,
    model: str = MODEL,
    bank_ids: str = "",
):
    initialize(
        host,
        port,
        model,
        bank_ids,
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    fire.Fire(main)
