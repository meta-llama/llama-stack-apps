set -a
source docqa_env
echo $DOC_PATH
# Run GPU version of ollama docker
if [ "$USE_GPU_FOR_DOC_INGESTION" = true ]; then
    echo "Running with GPU"
    docker compose --file compose-gpu.yaml up
else
# Run CPU version of ollama docker
    echo "Running with CPU only"
    docker compose --file compose-cpu.yaml up
fi
