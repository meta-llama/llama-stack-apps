# Interior Design Assistant

The Interior Design Assistant app is an example of how to leverage Llama3.2-Vision models and the LlamaStack client to do multimodal understanding and RAG.

![Example Screenshot](https://github.com/meta-llama/llama-stack-apps/blob/main/examples/interior_design_assistant/resources/demo.png)

The app has 3 interesting parts:

1. Multimodal understanding of the provided image. This agent describes the image in detail and identifies major objects.
2. Multimodal understanding for analyzing the theme, architecture, design of the image and suggest alternatives that might fit well in the overall theme.
3. A RAG element that can take suggestion from [2] and find relevant items from a catalog ( or bank )

## Installtion
1. Git clone this repo
```
git clone https://github.com/meta-llama/llama-stack-apps.git
```
2. Start new env
```
conda create -n agentstore python=3.10
conda activate agentstore
cd <path/to/llama-stack-apps>
pip install -r requirements.txt
```

## Starting the app

1. Start your favorite Llama Stack distro (`llama stack run ...`)
2. Run the app script
```
# Script to test the api
PYTHONPATH=. python examples/interior_design_assistant/api.py localhost 5000 examples/interior_design_assistant/resources/documents/ examples/interior_design_assistant/resources/images/fireplaces

# Start gradio app
PYTHONPATH=. python examples/interior_design_assistant/app.py

```
The script takes 4 args - host and port where the distro is running. Path for documents which are used to build and index (on the fly) and path to the image index. Both are added to resources/ directory for ease of start.


## Generating descriptions

The app relies on docuemnts with image descriptions or Retrieval. We also provide utilities to generate these descriptions as an example.

```
# PYTHONPATH=. python examples/interior_design_assistant/generate_descriptions.py host port image_dir output_document_dir

PYTHONPATH=. python examples/interior_design_assistant/generate_descriptions.py localhost 5000 examples/interior_design_assistant/resources/images/fireplaces/ examples/interior_design_assistant/resources/documents
```
