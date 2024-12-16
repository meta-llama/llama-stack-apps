# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import abc
from typing import Optional


class BaseUser(abc.ABC):
    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, content: str) -> str:
        raise NotImplementedError


class HumanUser(BaseUser):
    def reset(self, instruction: str) -> str:
        return input(f"{instruction}\n")

    def step(self, content: str) -> str:
        return input(f"{content}\n")


# class LLMUserSimulationEnv(BaseUserSimulationEnv):
#     def __init__(self, model: str, provider: str) -> None:
#         super().__init__()
#         self.messages: List[Dict[str, Any]] = []
#         self.model = model
#         self.provider = provider
#         self.total_cost = 0.0
#         self.reset()

#     def generate_next_message(self, messages: List[Dict[str, Any]]) -> str:
#         res = completion(
#             model=self.model, custom_llm_provider=self.provider, messages=messages
#         )
#         message = res.choices[0].message
#         self.messages.append(message.model_dump())
#         self.total_cost = res._hidden_params["response_cost"]
#         return message.content

#     def build_system_prompt(self, instruction: Optional[str]) -> str:
#         instruction_display = (
#             ("\n\nInstruction: " + instruction + "\n")
#             if instruction is not None
#             else ""
#         )
#         return f"""You are a user interacting with an agent.{instruction_display}
# Rules:
# - Just generate one line at a time to simulate the user's message.
# - Do not give away all the instruction at once. Only provide the information that is necessary for the current step.
# - Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.
# - If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.
# - Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.
# - Try to make the conversation as natural as possible, and stick to the personalities in the instruction."""

#     def reset(self, instruction: Optional[str] = None) -> str:
#         self.messages = [
#             {
#                 "role": "system",
#                 "content": self.build_system_prompt(instruction=instruction),
#             },
#             {"role": "user", "content": "Hi! How can I help you today?"},
#         ]
#         return self.generate_next_message(self.messages)

#     def step(self, content: str) -> str:
#         self.messages.append({"role": "user", "content": content})
#         return self.generate_next_message(self.messages)

#     def get_total_cost(self) -> float:
#         return self.total_cost
