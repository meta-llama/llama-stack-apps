# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from llama_stack_client.types.agent_create_params import AgentConfig

MEMORY_BANK_ID = "rag_agent_docs"

MEMORY_BANK_PARAMS = {
    "embedding_model": "all-MiniLM-L6-v2",
    "chunk_size_in_tokens": 512,
    "overlap_size_in_tokens": 64,
}

AGENT_CONFIG = AgentConfig(
    model="Llama3.1-405B-Instruct",
    instructions="You are a helpful assistant",
    sampling_params={
        "strategy": "greedy",
        "temperature": 1.0,
        "top_p": 0.9,
    },
    tools=[
        {
            "type": "memory",
            "memory_bank_configs": [{"bank_id": MEMORY_BANK_ID, "type": "vector"}],
            "query_generator_config": {"type": "default", "sep": " "},
            "max_tokens_in_context": 4096,
            "max_chunks": 10,
        }
    ],
    tool_choice="auto",
    tool_prompt_format="json",
    input_shields=[],
    output_shields=[],
    enable_session_persistence=False,
)
