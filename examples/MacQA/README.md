## MacQA

This is an end-to-end Retrieval Augmented Generation (RAG) App leveraging llama-stack that handles the logic for ingesting documents, storing them in a vector database and providing an inference interface.


### Prerequisite:

**Install ollama**: This app use ollama to run inference, please follow [ollama's download instruction](https://ollama.com/download) to install Ollama.

### How to run:

1. Open ollama software.
2. Run `./MacQA` in terminal
3. Open `http://localhost:7861/`, then type the path of data folder and choose the model for the Ollama inference.
4. Wait for the setup to be ready and click `Chat` tab to start chating to this app.

### How to build:

1. Run `pip install -r requirements.txt` to install pypi packages.
2. Run `python MacQA.py` to make sure everything works.
3. Compile MacQA.py by `pyinstaller MacQA.spec`, the binary should be in `./dist/MacQA`
