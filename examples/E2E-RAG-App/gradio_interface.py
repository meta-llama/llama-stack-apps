import asyncio
import json
import os
import uuid
from typing import List, Optional

import gradio as gr
import requests
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document

# Load environment variables
load_dotenv()


class LlamaChatInterface:
    def __init__(self, host: str, port: int, docs_dir: str):
        self.host = host
        self.port = port
        self.docs_dir = docs_dir
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "test_bank_666"

    async def initialize_system(self):
        """Initialize the entire system including memory bank and agent."""
        await self.setup_memory_bank()
        await self.load_documents()
        await self.initialize_agent()

    def is_memory_bank_present(self, target_identifier):
        """Checks if a memory bank exists."""
        return any(
            bank.identifier == target_identifier
            for bank in self.client.memory_banks.list()
        )

    async def setup_memory_bank(self):
        """Set up the memory bank if it doesn't exist."""
        providers = self.client.providers.list()
        provider_id = providers["memory"][0].provider_id
        memorybank_list = self.client.memory_banks.list()
        # for bank in memorybank_list:
        #     self.client.memory_banks.unregister(memory_bank_id=bank.identifier)
        if not self.is_memory_bank_present(self.memory_bank_id):
            memory_bank = self.client.memory_banks.register(
                memory_bank_id=self.memory_bank_id,
                params={
                    "embedding_model": "all-MiniLM-L6-v2",
                    "chunk_size_in_tokens": 100,
                    "overlap_size_in_tokens": 10,
                },
                provider_id=provider_id,
            )
            print(f"Memory bank registered: {memory_bank}")

    async def load_documents(self):
        """Load documents from the specified directory into memory bank."""
        documents = []
        for filename in os.listdir(self.docs_dir):
            if filename.endswith((".txt", ".md")):
                file_path = os.path.join(self.docs_dir, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    document = Document(
                        document_id=filename,
                        content=content,
                        mime_type="text/plain",
                        metadata={"filename": filename},
                    )
                    documents.append(document)

        if documents:
            self.client.memory.insert(
                bank_id=self.memory_bank_id,
                documents=documents,
            )
            print(f"Loaded {len(documents)} documents from {self.docs_dir}")

    async def initialize_agent(self):
        """Initialize the agent with model registration and configuration."""
        model_name = "Llama3.2-3B-Instruct"

        # Agent configuration
        agent_config = AgentConfig(
            model=model_name,
            instructions="You are a helpful assistant that can answer questions based on provided documents.",
            sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
            tools=[
                {
                    "type": "memory",
                    "memory_bank_configs": [
                        {"bank_id": self.memory_bank_id, "type": "vector"}
                    ],
                    "query_generator_config": {"type": "default", "sep": " "},
                    "max_tokens_in_context": 300,
                    "max_chunks": 5,
                }
            ],
            tool_choice="auto",
            tool_prompt_format="json",
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = self.agent.create_session(f"session-{uuid.uuid4()}")

    async def chat(self, message: str, history: List[List[str]]) -> str:
        """Process a chat message and return the response."""
        if self.agent is None:
            await self.initialize_system()

        response = self.agent.create_turn(
            messages=[{"role": "user", "content": message}], session_id=self.session_id
        )

        # Collect the response using EventLogger
        full_response = ""
        async for log in EventLogger().log(response):
            if hasattr(log, "content"):
                full_response += log.content

        return full_response


def create_gradio_interface(
    host: str = "localhost", port: int = 5000, docs_dir: str = "./docs"
):
    # Initialize the chat interface
    chat_interface = LlamaChatInterface(host, port, docs_dir)

    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# LlamaStack Chat")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(
            label="Message", placeholder="Type your message here...", show_label=False
        )
        with gr.Row():
            submit = gr.Button("Send")
            clear = gr.Button("Clear")

        gr.Examples(
            examples=[
                "What topics are covered in the documents?",
                "Can you summarize the main points?",
                "Tell me more about specific details in the text.",
            ],
            inputs=msg,
        )

        async def respond(message, chat_history):
            bot_message = await chat_interface.chat(message, chat_history)
            chat_history.append((message, bot_message))
            return "", chat_history

        def clear_chat():
            return None

        # Set up event handlers
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        submit.click(respond, [msg, chatbot], [msg, chatbot])
        clear.click(clear_chat, None, chatbot)

    return interface


if __name__ == "__main__":
    # Create and launch the Gradio interface
    interface = create_gradio_interface(
        docs_dir="/root/rag_data"
    )  # Specify your docs directory here
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)
