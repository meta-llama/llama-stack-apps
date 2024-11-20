## E2E-RAG-App

This is an End to End RAG App leveraging llama-stack that handles the logic for ingesting documents, storing them in a vector db and providing an inference interface.

We share the details of how to run first and then an outline of how it works:

## Prerequisite:

Install docker: Check [this doc for Mac](https://docs.docker.com/desktop/setup/install/mac-install/), [this doc for Windows](https://docs.docker.com/desktop/setup/install/windows-install/) and this [instruction for Linux](https://docs.docker.com/engine/install/).

For Mac and Windows users, you need to start the Docker app manually after installation.

## How to run:

1. We have main config `RAG_service.json` inside of the docker folder, please change `model_name` and `document_path` accordingly, for example:

```yaml
{
  "model_name": "llama3.2:1b-instruct-fp16",
  "document_path": "${HOME}/work/llama-stack-apps/examples/E2E-RAG-App/example_data"
}
```

2. Inside of docker folder, `run_RAG.sh` is the main script that can create `.env` file for compose.yaml and then actually start the `docker compose` process to launch all the pipelines in our dockers. `compose.yaml` is the main docker yaml that specifies all the mount option and docker configs, change the mounts if needed.

```bash
cd docker
bash run_RAG.sh
```

> [!TIP]
> You can check the status of dockers by typing `docker ps` on another terminal.

3. Ollama docker will start and this docker will pull and run the llama model specified. The `ollama_start.sh` control the Ollama docker startup behavior, change it if needed.

> [!TIP]
> On anther terminal, you can log into the docker and check which model has been hosted, by following code:

```bash
docker exec -it docker-ollama-1 bash
ollama ps
```

> Check more about Ollama instruction [here](https://github.com/ollama/ollama)

4. ChromaDB docker will also start. This docker will host the chroma database that can interact with llama-stack.

5. Lastly, Llama-stack docker will start. The `llama_stack_start.sh` control the docker startup behavior, change it if needed. It should be able to run llama-stack server based on the  `llama_stack_run.yaml` config. Once the server is ready, then it will run the `gradio_interface.py`.

6. `gradio_interface.py` will show a public link. You can access the gradio UI by putting this link to the browser. Then you can start your chat in the gradio web page.


All of the steps are run using a single-step via docker script.

Overview of how it works:
1. We use [docling](https://github.com/DS4SD/docling) framework for handling multiple file input formats (PDF, PPTX, DOCX)
2. If you are using a GPU inference machine, we have an option to use `Llama-3.2-11B-Vision` to caption images in the documents, on CPU machine this step is skipped
3. Once ingested, we use a llama-stack distribution running chroma-db and `Llama-3.2-3B-Instruct` to ingest chunks into a memory_bank
4. Once the vectordb is created, we then use llama-stack with the `Llama-3.2-3B-Instruct` to chat with the model.

![RAG_workflow](./RAG_workflow.jpg)
