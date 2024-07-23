set -o xtrace

conda deactivate
with-proxy conda create -n llagsys10 python=3.10
conda activate llagsys10

with-proxy pip install -r requirements.txt

pre-commit install

mkdir ~/models/prompt_guard/ && cd $_ && manifold --prod-use-cython-client getr llama3_agent_system/tree/checkpoints/prompt_guard

mkdir ~/models/llama_guard_v3/ && cd $_ && manifold --prod-use-cython-client getr llama3_agent_system/tree/checkpoints/llama_guard_v3

mkdir ~/models/mm_llama3_70b_19_06_2024_rlhf_v5 && cd $_ && manifold --prod-use-cython-client getr llama3_agent_system/tree/checkpoints/mm_llama3_70b_19_06_2024_rlhf_v5

mkdir ~/models/Meta-Llama-3.1-8B-Instruct-20240710150000/ && cd $_ && manifold --prod-use-cython-client getr llama3_agent_system/tree/checkpoints/Meta-Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-20240710150000
