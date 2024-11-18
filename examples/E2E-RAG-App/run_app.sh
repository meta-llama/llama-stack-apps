#!/bin/bash

echo "Running 01_ingest_files.py..."
python 01_ingest_files.py
if [ $? -ne 0 ]; then
  echo "Error running 01_ingest_files.py"
  exit 1
fi

echo "Running 02_caption_outputs.py..."
python 02_caption_outputs.py
if [ $? -ne 0 ]; then
  echo "Error running 02_caption_outputs.py"
  exit 1
fi

echo "Running ollama_main.py..."
python ollama_main.py localhost 5000 ./data/output/
if [ $? -ne 0 ]; then
  echo "Error running ollama_main.py"
  exit 1
fi

echo "All scripts ran successfully!"
