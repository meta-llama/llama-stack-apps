#!/bin/bash
echo "-------------start to serve------------"
/usr/bin/ollama serve > /dev/null 2>&1
sleep 10 
echo "pulling ollama model: $MODEL_NAME"
/usr/bin/ollama pull $MODEL_NAME
echo "Running ollama model: $MODEL_NAME"
/usr/bin/ollama run $MODEL_NAME 
while :; do sleep 2073600; done
