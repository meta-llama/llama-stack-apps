# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import fire
from termcolor import cprint

from .agents import get_retail_agent
from .tasks.retail.tasks_dev import TASKS_DEV
from .users.users import HumanUser


def main():
    max_steps = 30
    agent = get_retail_agent()

    user = HumanUser()
    task = TASKS_DEV[0]
    instruction = task.instruction
    last_agent_response = ""

    for i in range(max_steps):
        print("BEFORE", agent.env.data["orders"]["#W5442520"])
        if i == 0:
            user_input = user.reset(instruction)
            cprint(f"(Step {i}) User: {user_input}", "grey")
        else:
            user_input = user.step(last_agent_response)
            cprint(f"(Step {i}) User: {user_input}", "grey")

        # pass user input to agent
        agent_response = agent.step(
            {
                "role": "user",
                "content": user_input,
            }
        )
        cprint(f"(Step {i}) Agent: {agent_response}", "grey")
        last_agent_response = agent_response

        # Evaluate if the agent has complete the task
        print(task.tool_calls)
        print("AFTER", agent.env.data["orders"]["#W5442520"])


if __name__ == "__main__":
    fire.Fire(main)
