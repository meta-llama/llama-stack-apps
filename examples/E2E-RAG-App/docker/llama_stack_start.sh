
sleep 45
#echo "starting to install llama-stack"

pip install gradio

echo "starting the llama-stack server"
python -m llama_stack.distribution.server.server --yaml-config /root/my-run.yaml --disable-ipv6&
sleep 36000000
echo "running the RAG app"
python /root/E2E-RAG-App/gradio_interface.py
#python /root/E2E-RAG-App/ollama_main.py localhost 5000
