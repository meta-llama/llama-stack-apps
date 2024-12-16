# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import fire
from termcolor import cprint

from .agents import get_retail_agent
from .agents.retail.tools import ALL_TOOLS
from .base_env import BaseEnv
from .tasks.retail.tasks_dev import TASKS_DEV
from .users.users import HumanUser


def main():
    max_steps = 30
    agent = get_retail_agent()

    user = HumanUser()
    task = TASKS_DEV[0]
    instruction = task.instruction
    last_agent_response = ""

    gt_env = BaseEnv("retail")
    gt_tools = [tool(gt_env) for tool in ALL_TOOLS]
    gt_tools_map = {tool.get_name(): tool for tool in gt_tools}

    for i in range(max_steps):
        if i == 0:
            user_input = user.reset(instruction)
            cprint(f"(Step {i}) User: {user_input}", "grey")
        else:
            user_input = user.step(last_agent_response)
            cprint(f"(Step {i}) User: {user_input}", "grey")

        if user_input == "### STOP":
            break

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
    for gt_tool in task.tool_calls:
        gt_tools_map[gt_tool.tool_name].run_impl(**gt_tool.arguments)

    if gt_env == agent.env:
        cprint("✓ Task Successfully Completed!", "cyan")
    else:
        cprint("✗ Task Failed!", "red")

    gt_env.reset()
    agent.env.reset()


if __name__ == "__main__":
    fire.Fire(main)
