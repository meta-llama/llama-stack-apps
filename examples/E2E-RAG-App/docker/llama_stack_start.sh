#!/bin/bash
# Sleep for 45 seconds
sleep 45
# Print a message indicating the start of llama-stack docker
echo "-----starting to llama-stack docker now---------"
# Install required packages
pip install gradio

pip install -U llama-stack
# Check if GPU is enabled and run ingest files script accordingly
if [ "$USE_GPU" = true ]; then
  pip install docling
  python /root/E2E-RAG-App/01_ingest_files.py --input_dir /root/rag_data/
fi
# Print a message indicating the start of llama-stack server
echo "starting the llama-stack server"
# Run llama-stack server with specified config and disable ipv6
python -m llama_stack.distribution.server.server --yaml-config /root/my-run.yaml --disable-ipv6 &
# Sleep for 30 seconds
sleep 30
# Print a message indicating the start of RAG app
echo "---------running the RAG app--------------"
# Run RAG app
python /root/E2E-RAG-App/app.py
