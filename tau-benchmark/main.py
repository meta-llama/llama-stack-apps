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

from .users.users import HumanUser, SimulatedUser

MAX_STEPS = 30
STOP_TOKEN = "###STOP###"


def run_single_task(
    task_idx: int = 0, user_type: str = "simulated", verbose: bool = True
):
    max_steps = MAX_STEPS
    agent = get_retail_agent()

    task = TASKS_DEV[task_idx]
    instruction = task.instruction
    user = SimulatedUser(instruction)
    if user_type == "human":
        user = HumanUser()
    elif user_type == "simulated":
        user = SimulatedUser(instruction)
    else:
        raise ValueError(f"Invalid user type: {user_type}")

    last_agent_response = ""

    gt_env = BaseEnv("retail")
    gt_tools = [tool(gt_env) for tool in ALL_TOOLS]
    gt_tools_map = {tool.get_name(): tool for tool in gt_tools}

    for i in range(max_steps):
        if i == 0:
            user_input = user.reset(instruction)
        else:
            user_input = user.step(last_agent_response)

        cprint(f"(Step {i}) 👤 User: ", "white", end="", attrs=["bold"])
        cprint(f" {user_input}", "white")

        if STOP_TOKEN in user_input:
            break

        # pass user input to agent
        agent_response = agent.step(
            {
                "role": "user",
                "content": user_input,
            },
            verbose=verbose,
        )
        cprint(f"(Step {i}) 🤖 Agent: ", "light_blue", end="", attrs=["bold"])
        cprint(f" {agent_response.content}", "light_blue")
        last_agent_response = agent_response.content
        cprint("-" * 20, "grey")

    # Evaluate if the agent has complete the tasks
    for gt_tool in task.tool_calls:
        print(gt_tool)
        gt_tools_map[gt_tool.tool_name].run_impl(**gt_tool.arguments)

    if gt_env == agent.env:
        cprint(f"✓ Task Successfully completed in {i} steps", "cyan", attrs=["bold"])
    else:
        cprint(f"✗ Task Failed in {i} steps", "red", attrs=["bold"])

    gt_env.reset()
    agent.env.reset()


def main(task_idx: int = 0, user_type: str = "simulated", verbose: bool = True):
    if task_idx < 0:
        for task_idx in range(len(TASKS_DEV)):
            run_single_task(task_idx, user_type, verbose)
    else:
        run_single_task(task_idx, user_type, verbose)


if __name__ == "__main__":
    fire.Fire(main)