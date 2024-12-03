#!/bin/bash

echo "DOC_PATH=$(jq -r '.document_path' ./RAG_service.json)" > .env
echo "MODEL_NAME=$(jq -r '.model_name' ./RAG_service.json)" >> .env
echo "HOST=$(jq -r '.host' ./RAG_service.json)" >> .env
echo "PORT=$(jq -r '.port' ./RAG_service.json)" >> .env
echo "CHROMA_PORT=$(jq -r '.chroma_port' ./RAG_service.json)" >> .env
echo "GRADIO_SERVER_PORT=$(jq -r '.gradio_server_port' ./RAG_service.json)" >> .env
echo "USE_GPU=$(jq -r '.use_gpu' ./RAG_service.json)" >> .env
# Run GPU version of ollama docker
if [ "$(jq -r '.use_gpu' ./RAG_service.json)" = true ]; then
    echo "Running with GPU"
    docker compose --file compose-gpu.yaml up
else
# Run CPU version of ollama docker
    echo "Running with CPU only"
    docker compose --file compose-cpu.yaml up
fi
