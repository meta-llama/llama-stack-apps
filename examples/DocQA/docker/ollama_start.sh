#!/bin/bash
echo "-------------start to serve------------"
OLLAMA_HOST=127.0.0.1:14343 /usr/bin/ollama  serve &
echo "Running ollama model: $MODEL_NAME"
sleep 5
OLLAMA_HOST=127.0.0.1:14343 /usr/bin/ollama run $MODEL_NAME
