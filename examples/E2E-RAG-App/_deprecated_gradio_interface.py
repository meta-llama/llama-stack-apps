import asyncio
import json
import os
from typing import List, Optional
import gradio as gr
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.types.agent_create_params import AgentConfig

# Load environment variables
load_dotenv()

class LlamaChatInterface:
    def __init__(self, host: str, port: int):
        self.client = LlamaStackClient(base_url=f"http://{host}:{port}")
        self.agent = None
        self.session_id = None
        self.memory_bank_id = "test_bank_3"
        
    async def initialize_agent(self):
        # Model registration
        model_name = "Llama3.2-3B-Instruct"
        
        # Agent configuration
        agent_config = AgentConfig(
            model=model_name,
            instructions="You are a helpful assistant that can answer questions based on provided documents.",
            sampling_params={"strategy": "greedy", "temperature": 1.0, "top_p": 0.9},
            tools=[
                {
                    "type": "memory",
                    "memory_bank_configs": [{"bank_id": self.memory_bank_id, "type": "vector"}],
                    "query_generator_config": {"type": "default", "sep": " "},
                    "max_tokens_in_context": 4096,
                    "max_chunks": 10,
                }
            ],
            tool_choice="auto",
            tool_prompt_format="json",
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = str(uuid.uuid4())

    async def chat(self, message: str, history: List[List[str]]) -> str:
        if self.agent is None:
            await self.initialize_agent()
            
        response = await self.agent.create_turn(
            messages=[{"role": "user", "content": message}],
            session_id=self.session_id
        )
        
        # Extract the assistant's response from the response object
        # Note: You might need to adjust this based on the actual response structure
        assistant_message = ""
        async for chunk in response:
            if hasattr(chunk, 'delta') and chunk.delta:
                assistant_message += chunk.delta
        
        return assistant_message

def create_gradio_interface(host: str = "localhost", port: int = 8000):
    # Initialize the chat interface
    chat_interface = LlamaChatInterface(host, port)
    
    # Create the Gradio interface
    iface = gr.ChatInterface(
        fn=chat_interface.chat,
        title="LlamaStack Chat",
        description="Chat with your documents using LlamaStack",
        examples=[
            ["What topics are covered in the documents?"],
            ["Can you summarize the main points?"],
            ["Tell me more about specific details in the text."],
        ],
        theme=gr.themes.Soft()
    )
    
    return iface

if __name__ == "__main__":
    # Create and launch the Gradio interface
    iface = create_gradio_interface()
    iface.launch(server_name="0.0.0.0", server_port=7860)
