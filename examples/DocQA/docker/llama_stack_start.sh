#!/bin/bash

sleep 45
echo "-----starting to llama-stack docker now---------"
pip install gradio

if [ "$USE_GPU_FOR_DOC_INGESTION" = true ]; then
  echo "Using GPU to ingest files"
  pip install docling
  python /root/DocQA/scripts/ingest_files.py --input_dir /root/rag_data/
fi
echo "starting the llama-stack server"
python -m llama_stack.distribution.server.server --yaml-config /root/my-run.yaml --disable-ipv6&
sleep 30
echo "---------running the RAG app--------------"
python /root/DocQA/app.py
