import os
import re
import subprocess
from multiprocessing import freeze_support
from typing import Generator, List, Optional
import base64
import mimetypes
import gradio as gr
from dotenv import load_dotenv

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient

from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.agent import Agent
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.memory_insert_params import Document

# Load environment variables
load_dotenv()

GRADIO_SERVER_PORT = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
# if USE_GPU_FOR_DOC_INGESTION, then the documents will be processed to output folder
CUSTOM_CSS = """
.context-block {
    font-size: 0.8em;
    border-left: 3px solid #e9ecef;
    margin: 0.5em 0;
    padding: 0.5em 1em;
    opacity: 0.85;
}

.context-title {
    font-size: 0.8em;
    color: #9ca3af;
    font-weight: 400;
    display: flex;
    align-items: center;
    gap: 0.5em;
    margin-bottom: 0.3em;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.context-title::before {
    content: "📄";
    font-size: 1em;
    opacity: 0.7;
}

.context-content {
    color: #6b7280;
    line-height: 1.4;
    font-weight: 400;
}

.inference-response {
    font-size: 1em;
    color: #111827;
    line-height: 1.5;
    margin-top: 1em;
}
"""
from pydantic import BaseModel,Field


class MetaData(BaseModel):
    '''Product description saved as metadata'''
    title: str = Field(..., title="Product Title", description="Title of the product")
    description: str = Field(..., title="Product Description", description="Description of the product")
    tags: list = Field([], title="Product Tags", description="Tags for SEO")
    primary_colors: list = Field([], title="Primary Colors", description="Primary colors of the product")


class LlamaChatInterface:
    def __init__(self):
        self.client = None

    def initialize_system(self, provider_name="ollama"):
        """Initialize the entire system including memory bank and agent."""
        # path_to_yaml = os.path.abspath(os.path.join(os.path.dirname(__file__), "llama_stack_run.yaml"))
        self.client = LlamaStackAsLibraryClient(provider_name)
        #print(type(self.client.async_client.config), self.client.async_client.config)

        # Disable scoring and eval by modifying the config
        self.client.async_client.config.apis = [
            "agents",
            "datasetio",
            "inference",
            "memory",
            "safety",
            "telemetry",
            "tool_runtime",
        ]
        # self.client.async_client.config.tool_groups = []
        del self.client.async_client.config.providers["scoring"]
        del self.client.async_client.config.providers["eval"]
        del self.client.async_client.config.providers["tool_runtime"]


        self.client.async_client.config.tool_groups = []
        # print(
        #     111, type(self.client.async_client.config), self.client.async_client.config
        # )

        self.client.initialize()

    # def initialize_agent(self):
    #     """Initialize the agent with model registration and configuration."""

    #     agent_config = AgentConfig(
    #         model=MODEL_NAME,
    #         instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
    #         toolgroups=[
    #             {
    #                 "name": "builtin::memory",
    #                 "args": {"memory_bank_ids": [self.memory_bank_id]},
    #             }
    #         ],
    #         enable_session_persistence=True,
    #     )
    #     self.agent = Agent(self.client, agent_config)
    #     self.session_id = self.agent.create_session(f"session-docqa")

    def get_metadata_from_image(self, image_path, custom_prompt):
        # Extract metadata from the image
        data_url = encode_image_to_data_url(image_path)
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = f"""
    Given the image of a product, provide the following information in English:
    - Product Title
    - Product Description
    - At least 7 Product Tags for SEO purposes
    - At most 3 primary Colors of the Product, excluding the background colors.
    - Do not include any information that is not relevant to the product or is not visible in the image.
    - MUST return the information in JSON format, with the keys "title", "description", "tags", "primary_colors".
    """
        message = {
            "role": "user",
            "content": [
                {"type": "image", "url": {"uri": data_url}},
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }

        response = self.client.inference.chat_completion(
            messages=[message],
            model_id=MODEL_NAME,
            # response_format={
            #     "type": "json_schema",
            #     "json_schema": MetaData.model_json_schema(),
            # },
        )

        return response.completion_message.content


def encode_image_to_data_url(file_path: str) -> str:
    """
    Encode an image file to a data URL.

    Args:
        file_path (str): Path to the image file

    Returns:
        str: Data URL string
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_string}"
def create_gradio_interface():
    chat_interface = LlamaChatInterface()
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as main_interface:
        gr.Markdown("# LlamaStack Image Tagging Demo")
        #image_input = gr.Image(type="pil", label="Upload image")
        image_input = gr.Image(label="Upload Image", type="filepath")
        custom_prompt = gr.Textbox(
            label="Custom Prompt", placeholder="Describe this image in detail."
        )

        generate_button = gr.Button("Generate")
        # Define the Gradio interface
        metadata_output = gr.Textbox(label="Ouput", interactive=False)
        generate_button.click(
            fn=chat_interface.get_metadata_from_image,
            inputs=[image_input, custom_prompt],
            outputs=metadata_output,
        )

    # Combine both interfaces
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        with gr.Tab("Setup", visible=True) as setup_tab:
            with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as initial_interface:
                gr.Markdown("## Use the following options to setup the chat interface")
                model_name_input = gr.Dropdown(
                    choices=[
                        "meta-llama/Llama-3.2-11B-Vision-Instruct",
                        "meta-llama/Llama-3.2-90B-Vision-Instruct",
                    ],
                    value="meta-llama/Llama-3.2-11B-Vision-Instruct",
                    label="Llama Model Name",
                )
                provider_name = gr.Dropdown(
                    choices=[
                        "together",
                        "fireworks",
                    ],
                    value="together",
                    label="Provider List",
                )
                api_key = gr.Textbox(label="Put your API key here")
                setup_button = gr.Button("Setup Interface")
                setup_output = gr.Textbox(label="Setup", interactive=False)

                # Function to handle the initial input and transition to the chat interface
                def setup_chat_interface(model_name, provider_name, api_key):
                    global MODEL_NAME
                    MODEL_NAME = model_name
                    if provider_name == "together":
                        os.environ["TOGETHER_API_KEY"] = api_key
                    elif provider_name == "fireworks":
                        os.environ["FIREWORKS_API_KEY"] = api_key
                    try:
                        print("Starting LlamaStack direct client...")
                        os.environ["INFERENCE_MODEL"] = model_name
                        print("Using model: ", model_name)
                        print("Using provider: ", provider_name)
                        chat_interface.initialize_system(provider_name)
                    except Exception as e:
                        print(f"Error: {e}")
                        raise gr.exceptions.Error(e)
                        return (f"Error: {e}",)
                    return (
                        f"Model {model_name} inference started using provider {provider_name} You can now go to Chat tab and start chatting!",
                    )

                setup_button.click(
                    setup_chat_interface,
                    inputs=[
                        model_name_input,
                        provider_name,
                        api_key,
                    ],
                    outputs=setup_output,
                )
        with gr.Tab("Chat", visible=True) as chat_tab:
            main_interface.render()
    return demo


if __name__ == "__main__":
    # Create and launch the Gradio interface
    freeze_support()
    interface = create_gradio_interface()
    interface.launch(
        server_name="localhost", server_port=GRADIO_SERVER_PORT, debug=True
    )
