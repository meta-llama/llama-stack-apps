#!/bin/bash
echo "-------------start to serve------------"
/usr/bin/ollama  serve &
echo "Running ollama model: $MODEL_NAME"
sleep 3
/usr/bin/ollama run $MODEL_NAME
