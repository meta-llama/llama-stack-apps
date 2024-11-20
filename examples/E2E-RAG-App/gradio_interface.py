import asyncio
import json
import os
import uuid
from queue import Queue
from threading import Thread
from typing import AsyncGenerator, Generator, List, Optional

import chromadb
import gradio as gr
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document

# Load environment variables
load_dotenv()

HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 5000))
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 6000))
DOCS_DIR = "/root/rag_data/output"
GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", 7861))
MODEL_NAME = os.getenv("MODEL_NAME", "Llama3.2-1B-Instruct")

# Custom CSS for enhanced styling
CUSTOM_CSS = """
.message-rag {
    font-size: 0.875rem !important;
    background-color: rgba(30, 41, 59, 0.5) !important;
    border-radius: 0.5rem !important;
    padding: 0.75rem !important;
    margin-bottom: 1rem !important;
    font-family: ui-monospace, monospace !important;
}

.message-rag-title {
    font-size: 0.75rem !important;
    color: #94a3b8 !important;
    margin-bottom: 0.25rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 0.5rem !important;
}

.message-rag-title::before {
    content: "📄" !important;
    font-size: 1rem !important;
}

.message-rag-content {
    color: #cbd5e1 !important;
}

.bot-message {
    font-size: 1rem !important;
    line-height: 1.5 !important;
}

.user-message {
    background-color: rgb(79, 70, 229) !important;
}
"""

class LlamaChatInterface:
    def __init__(self, host: str, port: int, chroma_port: int, docs_dir: str):
        self.host = host
        self.port = port
        self.docs_dir = docs_dir
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.chroma_client = chromadb.HttpClient(host=host, port=chroma_port)
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "test_bank_212"

    # ... [previous methods remain the same until chat_stream] ...

    def format_rag_context(self, context: str) -> str:
        """Format RAG context with custom styling."""
        return f"""<div class="message-rag">
            <div class="message-rag-title">Retrieved context from memory:</div>
            <div class="message-rag-content">{context}</div>
        </div>"""

    def chat_stream(
        self, message: str, history: List[List[str]]
    ) -> Generator[List[List[str]], None, None]:
        history = history or []
        history.append([message, ""])
        
        output_queue = Queue()

        def run_async():
            async def async_process():
                if self.agent is None:
                    await self.initialize_system()

                response = self.agent.create_turn(
                    messages=[{"role": "user", "content": message}],
                    session_id=self.session_id,
                )

                current_response = ""
                context_shown = False
                
                async for log in EventLogger().log(response):
                    log.print()
                    
                    # Handle RAG context differently
                    if hasattr(log, 'retrieved_context') and not context_shown:
                        context = self.format_rag_context(log.retrieved_context)
                        history[-1][1] = context + "\n"
                        context_shown = True
                        output_queue.put(history.copy())
                    
                    elif hasattr(log, 'content'):
                        current_response = log.content
                        # If we showed context before, append to it
                        if context_shown:
                            history[-1][1] = history[-1][1] + f'<div class="bot-message">{current_response}</div>'
                        else:
                            history[-1][1] = current_response
                        output_queue.put(history.copy())

                output_queue.put(None)

            asyncio.run(async_process())

        thread = Thread(target=run_async)
        thread.start()

        while True:
            item = output_queue.get()
            if item is None:
                break
            yield item

        thread.join()


def create_gradio_interface(
    host: str = HOST,
    port: int = PORT,
    chroma_port: int = CHROMA_PORT,
    docs_dir: str = DOCS_DIR,
):
    chat_interface = LlamaChatInterface(host, port, chroma_port, docs_dir)

    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as interface:
        gr.Markdown("# LlamaStack Chat")

        chatbot = gr.Chatbot(
            bubble_full_width=False,
            show_label=False,
            height=600,
            container=True,
            elem_classes={
                "user": "user-message",
                "bot": "bot-message"
            }
        )
        
        with gr.Row():
            with gr.Column(scale=20):
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
            with gr.Column(scale=1, min_width=100):
                submit = gr.Button("Send", variant="primary")
                
        with gr.Row():
            clear = gr.Button("Clear Chat")

        gr.Examples(
            examples=[
                "What topics are covered in the documents?",
                "Can you summarize the main points?",
                "Tell me more about specific details in the text.",
            ],
            inputs=msg,
        )

        def clear_chat():
            return [], ""

        submit_event = msg.submit(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=msg,
        )

        submit_click = submit.click(
            fn=chat_interface.chat_stream,
            inputs=[msg, chatbot],
            outputs=chatbot,
            queue=True,
        ).then(
            fn=lambda: "",
            outputs=msg,
        )

        clear.click(clear_chat, outputs=[chatbot, msg], queue=False)

        msg.submit(lambda: None, None, None, api_name=False)
        interface.load(fn=chat_interface.initialize_system)

    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name=HOST,
        server_port=GRADIO_SERVER_PORT,
        share=True,
        debug=True
    )