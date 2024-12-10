#!/bin/bash

MAX_WAIT_TIME=30

sleep 30 # Wait for ollama

if [ -z "${LLAMA_STACK_PORT}" ]; then
    echo "Error: LLAMA_STACK_PORT environment variable is required" >&2
    exit 1
fi

echo "Installing dependencies..."
pip install gradio

if [ "$USE_GPU_FOR_DOC_INGESTION" = true ]; then
  echo "Using GPU to ingest files"
  pip install docling
  python /root/DocQA/scripts/ingest_files.py --input_dir /root/rag_data/
fi


if [ $timeout -eq $MAX_WAIT_TIME ]; then
    echo "Error: Ollama model failed to load within ${MAX_WAIT_TIME} seconds"
    exit 1
fi

echo "Starting the llama-stack server on port ${LLAMA_STACK_PORT}..."
python -m llama_stack.distribution.server.server --yaml-config /root/my-run.yaml --disable-ipv6 --port ${LLAMA_STACK_PORT} &

timeout=0
echo "Waiting for Llama Stack to come online..."
until curl -s localhost:${LLAMA_STACK_PORT} > /dev/null || [ $timeout -eq $MAX_WAIT_TIME ]; do
    sleep 1
    ((timeout++))
done

if [ $timeout -eq $MAX_WAIT_TIME ]; then
    echo "Error: Server failed to start within ${MAX_WAIT_TIME} seconds"
    kill $LLAMA_PID
    exit 1
fi

echo "Llama Stack available; starting RAG app..."
exec python /root/DocQA/app.py
