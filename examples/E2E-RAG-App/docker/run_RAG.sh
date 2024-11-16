echo  "DOC_PATH=$(jq '.document_path' ./RAG_service.json)" > .env
echo  "MODEL_NAME=$(jq '.model_name' ./RAG_service.json)" >> .env
docker compose up
