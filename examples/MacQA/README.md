## MacQA

This is an end-to-end Retrieval Augmented Generation (RAG) App leveraging llama-stack that handles the logic for ingesting documents, storing them in a vector database and providing an inference interface.

We share the details of how to run first and then an outline of how it works:

### Prerequisite:

**Install ollama**: This app use ollama to run inference, please follow [ollama's download instruction](https://ollama.com/download) to install Ollama.

**Install pypi packages**: Run `pip install -r requirements.txt` to install other pypi packages. Restart terminal to make chromadb affective, according to [this issue](https://github.com/langchain-ai/langchain/issues/1387#issuecomment-1614233339)

### How to run:

Run `./dist/MacQA`

### How to build:

1. `pip install pyinstaller` and all the pipy package needed
2. Run `MacQA.py` to make sure everything runs
3. Compile MacQA.py by `pyinstaller MacQA.spec`
