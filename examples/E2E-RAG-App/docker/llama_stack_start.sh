
sleep 60
echo "starting to install llama-stack"

pip install -e /root/llama-stack
#pip install -U llama-stack
pip uninstall -y chromadb-client
pip uninstall -y chromadb 
pip install -U chromadb
echo "Installing llama-stack-client"
pip install llama-stack-client==0.0.50
echo "starting the llama-stack server"
python -m llama_stack.distribution.server.server --yaml_config /root/my-run.yaml&
sleep 30

echo "running the RAG app"
python /root/llama-stack-apps/examples/E2E-RAG-App/rag_main.py localhost 5000 /root/RAG_service.json

