
sleep 45
#echo "starting to install llama-stack"
#apt-get update
#apt-get install -y git
#pip install /root/llama-stack
#pip install git+https://github.com/meta-llama/llama-stack.git@2edfda97e9659155074269fc3b7e66d9bb2c57d4
#pip install tiktoken
#pip install --upgrade --no-deps --force-reinstall --index-url https://test.pypi.org/simple/ llama_stack==0.0.53rc1
pip install gradio
echo "Installing llama-stack-client"
pip install distro
pip install --index-url https://test.pypi.org/simple/ llama_stack_client==0.0.53rc2
#pip install git+https://github.com/meta-llama/llama-stack-client-python.git@f5a2391241eac03eea356b206469081688277d23
echo "starting the llama-stack server"
python -m llama_stack.distribution.server.server --yaml-config /root/my-run.yaml --disable-ipv6&
sleep 3600000000

echo "running the RAG app"
#python /root/E2E-RAG-App/gradio_interface.py
#python /root/E2E-RAG-App/ollama_main.py localhost 5000
