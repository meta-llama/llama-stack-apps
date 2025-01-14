#!/bin/bash

MAX_WAIT_TIME=30

sleep 30 # Wait for ollama


echo "Installing dependencies..."
pip install gradio

if [ "$USE_GPU_FOR_DOC_INGESTION" = true ]; then
  echo "Using GPU to ingest files"
  pip install docling
  python /root/DocQA/scripts/ingest_files.py --input_dir /root/rag_data/
fi


echo "Starting RAG app..."
exec python /root/DocQA/app.py
