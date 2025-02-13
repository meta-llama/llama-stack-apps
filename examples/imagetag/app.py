import base64
import json
import mimetypes
import os
import re
import shelve
import subprocess
from multiprocessing import freeze_support
from pathlib import Path
from typing import Generator, List, Optional

import gradio as gr
from dotenv import load_dotenv

from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
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
from pydantic import BaseModel, Field

from string import Template 
DEFAULT_METADATA_PROMPT = f"""
    Given the image of a product, provide the following information in English:
    - Product Title
    - Product Description
    - At least 7 Product Tags for SEO purposes
    - At most 3 primary Colors of the Product, excluding the background colors.
    - Do not include any information that is not relevant to the product or is not visible in the image.
    - MUST return the information in JSON format, with the keys "title", "description", "tags", "primary_colors".
    """

DEFAULT_REWRITE_PROMPT = Template("""
    You are a helpful assistant. Rewrite the user's query to include details from the item description.
    Item description: $item_description
    User query: $user_query
    Please rewrite the query to include relevant details from the item description
""")

DEFAULT_SEARCH_PROMPT = Template("""
    You are a helpful assistant. Given the user's query and provided documents, you can answer the user's question.
    User query: $user_query
    return your answer in json format, with the key "answer" and "source". 
    the "source" MUST only be the absolute file path of the document that relates to the answer so that we can open it.
""")
class MetaData(BaseModel):
    """Product description saved as metadata"""

    title: str = Field(..., title="Product Title", description="Title of the product")
    description: str = Field(
        ..., title="Product Description", description="Description of the product"
    )
    tags: list = Field([], title="Product Tags", description="Tags for SEO")
    primary_colors: list = Field(
        [], title="Primary Colors", description="Primary colors of the product"
    )


def encode_image_to_data(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def find_images_set(search_directory):
    # Common image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    # Add uppercase versions of extensions
    image_extensions.update({ext.upper() for ext in image_extensions})

    image_path = Path(search_directory)
    return [
        str(file) for file in image_path.rglob("*.*") if file.suffix in image_extensions
    ]


def data_url_from_image(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError("Could not determine MIME type of the file")

    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    data_url = f"data:{mime_type};base64,{encoded_string}"
    return data_url


class LlamaChatInterface:
    def __init__(self):
        self.client = None
        self.custom_prompt = None
        self.imagestore = None
        self.agent = None
        self.session_id = None
        self.vector_db_id = "imagestore_db"

    def setup_vector_dbs(self):
        """Set up the vector db if it doesn't exist."""
        providers = self.client.providers.list()
        vector_io_provider = [
            provider for provider in providers if provider.api == "vector_io"
        ]
        provider_id = vector_io_provider[0].provider_id
        vector_dbs = self.client.vector_dbs.list()
        print("vector_dbs: ", vector_dbs)
        # Check if vector_dbs exists by identifier
        if vector_dbs and any(
            bank.identifier == self.vector_db_id for bank in vector_dbs
        ):
            self.client.vector_dbs.unregister(self.vector_db_id)
            print(f"vector_dbs '{self.vector_db_id}' exists but replaced with new one.")
        self.client.vector_dbs.register(
            vector_db_id=self.vector_db_id,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
        )
        print(f"vector_dbs registered.")

    def initialize_system(self, provider_name="ollama"):
        """Initialize the entire system including memory bank and agent."""
        self.client = LlamaStackAsLibraryClient(provider_name)


        # Disable scoring and eval by modifying the config
        self.client.async_client.config.apis = [
            "agents",
            "datasetio",
            "inference",
            "vector_io",
            "safety",
            "telemetry",
            "tool_runtime",
        ]
        # self.client.async_client.config.tool_groups = []
        del self.client.async_client.config.providers["scoring"]
        del self.client.async_client.config.providers["eval"]

        # only enable rag-runtime
        tool_groups = []
        for provider in self.client.async_client.config.tool_groups:
            if provider.provider_id == "rag-runtime":
                tool_groups.append(provider)
        self.client.async_client.config.tool_groups = tool_groups

        self.client.initialize()
        self.setup_vector_dbs()
        self.initialize_agent()


    def get_metadata_from_image(self, image_path, prompt):
        assert prompt is not None, "Prompt is not provided"
        assert len(prompt) > 0, "Prompt is empty"
        print(f"Prompt: {prompt}")
        message = {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": {
                    "url": {
                        # TODO: Replace with Github based URI to resources/sample1.jpg
                        "uri": data_url_from_image(image_path)
                        #"uri": "https://www.healthypawspetinsurance.com/Images/V3/DogAndPuppyInsurance/Dog_CTA_Desktop_HeroImage.jpg"
                    },
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ],
        }
        response = self.client.inference.chat_completion(
            model_id=MODEL_NAME,
            messages=[message],
            stream=False
        )
        metadata = response.completion_message.content.lower().strip()
        return metadata

    def process_folder(self, folder_path):
        if os.path.exists(folder_path + "/imagestore.json"):
            print(
                f"Found imagestore.json in {folder_path}, skipping images that have already been processed"
            )
            with open(folder_path + "/imagestore.json", "r") as file:
                self.imagestore = json.load(file)
            if self.imagestore.get("main_path") != folder_path:
                print(
                    f"Found imagestore.json in {folder_path}, but it is not for the current folder {folder_path}, skipping images that have already been processed"
                )
                return f"Found imagestore.json in {folder_path}, but it is not for the current folder {folder_path}, skipping images that have already been processed"
        else:
            self.imagestore = {}
            self.imagestore["main_path"] = folder_path
        image_files = find_images_set(folder_path)
        for image_path in image_files:
            # image_path = os.path.join(folder_path, file)
            if self.imagestore.get(image_path) is not None:
                print(f"Skipping {image_path} because it has already been processed")
                continue
            print(f"Processing image: {image_path}")
            # Extract metadata from the image
            if self.custom_prompt:
                prompt = self.custom_prompt
            else:
                prompt = DEFAULT_METADATA_PROMPT
            try:
                metadata = self.get_metadata_from_image(image_path, prompt)
                print(f"Metadata for {image_path}: {metadata}")
            except Exception as e:
                print(f"Error processing image: {image_path}")
                print(f"Error: {e}")
                continue
            self.imagestore[image_path] = metadata
        with open(folder_path + "/imagestore.json", "w") as file:
            json.dump(self.imagestore, file)
        print(f"Processed Selected folder: {folder_path}")


    def set_custom_prompt(self, custom_prompt):
        self.custom_prompt = custom_prompt

    def initialize_agent(self):
        """Initialize the agent with model registration and configuration."""

        agent_config = AgentConfig(
            model=MODEL_NAME,
            instructions="You are a helpful assistant that can answer questions based on provided documents. Return your answer short and concise, less than 50 words.",
            toolgroups=[
                {
                    "name": "builtin::rag",
                    "args": {"vector_db_ids": [self.vector_db_id]},
                }
            ],
            enable_session_persistence=True,
        )
        self.agent = Agent(self.client, agent_config)
        self.session_id = self.agent.create_session(f"session-docqa")

    def create_rag_from_imagestore(self):
        """Load documents from the specified directory into vector db."""
        assert self.imagestore is not None, "Imagestore is not initialized"
        documents = []
        for filename, content in self.imagestore.items():
            document = Document(
                document_id=filename,
                content=content,
                mime_type="text/plain",
                metadata={"filename": filename},
            )
            documents.append(document)

        if documents:
            self.client.tool_runtime.rag_tool.insert(
                documents=documents,
                vector_db_id=self.vector_db_id,
                chunk_size_in_tokens=512,
            )
            print(f"Loaded {len(documents)} metadata from imagestore")

    def rewrite_query(self, original_query, metadata):
        print(f"Rewriting query: {original_query}")
        messages = [
            {"role": "user", "content": DEFAULT_REWRITE_PROMPT.substitute(item_description=metadata,user_query=original_query)},
        ]

        try:
            response = self.client.inference.chat_completion(
                model=MODEL_NAME, messages=messages
            )
            rewritten_query = response.completion_message.content
            print(f"Rewritten query: {rewritten_query}")
            return rewritten_query
        except Exception as e:
            print(f"Error rewriting query: {e}")
            return original_query

    def search_database(self, query, image_path=None):
        if image_path:
            metadata = self.get_metadata_from_image(image_path, DEFAULT_METADATA_PROMPT)
            query = self.rewrite_query(query, metadata)
        response = self.agent.create_turn(
                messages=[{"role": "user", "content": DEFAULT_SEARCH_PROMPT.substitute(user_query=query)}],
                session_id=self.session_id,
            )
        logs = [str(log) for log in EventLogger().log(response) if log is not None]
        logs_str = "".join(logs).split("inference>")[-1]
        print(f"Search Results: {logs_str}")
        source = []
        try:
            response = json.loads(logs_str.strip())
            if "answer" in response:
                anwser = response["answer"]
            if "source" in response:
                source = response["source"]
                assert os.path.exists(response["source"]), "Source file does not exist"
                assert response["source"] in self.imagestore, "Source file is not in imagestore"
                # return answer to textbox and (source, description) to display in gallery 
                return anwser, [(source,self.imagestore[source])]
        except Exception as e:
            print(f"Error parsing response: {e}")
            return logs_str, source
        return logs_str, source


def create_gradio_interface():
    chat_interface = LlamaChatInterface()
    with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS) as main_interface:
        with gr.Row():
            query_input = gr.Textbox(label="Enter your query")
            response_output = gr.Textbox(label="Response")
        with gr.Row():
            image_input = gr.Image(label="Select an image")
            image_output = gr.Gallery(label="Retrieved Images")
        with gr.Row():
            search_button = gr.Button("Search")
        search_button.click(
            fn=chat_interface.search_database,
            inputs=[query_input, image_input],
            outputs=[response_output, image_output],
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
                # folder_input = gr.FileExplorer(
                #     glob="**/",
                #     label="Select a folder",
                #     interactive=True,  # Show only directories
                # )
                folder_input = gr.Textbox(label="Select a Folder Path with images")
                # custom_prompt = gr.Textbox(
                #     label="Custom Prompt", placeholder="Describe this image in detail."
                # )
                # if custom_prompt.value:
                #     chat_interface.set_custom_prompt(custom_prompt.value)
                setup_button = gr.Button("Setup Interface")
                setup_output = gr.Textbox(label="Setup", interactive=False)

                # Function to handle the initial input and transition to the chat interface
                def setup_chat_interface(
                    model_name, provider_name, api_key, folder_input
                ):
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
                        chat_interface.process_folder(folder_input)
                        chat_interface.create_rag_from_imagestore()
                    except Exception as e:
                        print(f"Error: {e}")
                        raise gr.exceptions.Error(e)
                        return (f"Error: {e}",)
                    return (
                        f"Model {model_name} inference started using provider {provider_name} You can now go to Chat tab and start chatting!",
                    )

                setup_button.click(
                    setup_chat_interface,
                    inputs=[model_name_input, provider_name, api_key, folder_input],
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
