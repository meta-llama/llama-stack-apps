
sleep 90
echo "starting to install llama-stack"
apt-get update
apt-get install -y git
#pip install /root/llama-stack
pip install git+https://github.com/meta-llama/llama-stack.git@2edfda97e9659155074269fc3b7e66d9bb2c57d4
pip uninstall -y chromadb-client
pip uninstall -y chromadb
pip install -U chromadb
echo "Installing llama-stack-client"
pip install llama-stack-client==0.0.50
echo "starting the llama-stack server"
python -m llama_stack.distribution.server.server --yaml_config /root/my-run.yaml&
sleep 30

echo "running the RAG app"
python /root/E2E-RAG-App/ollama_main.py localhost 5000 /root/RAG_service.json
