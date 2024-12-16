# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import fire
from termcolor import cprint

from .agents.retail_agent import RetailAgent
from .users.users import HumanUser


def main():
    max_steps = 30
    agent = RetailAgent()

    user = HumanUser()
    instruction = "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. If there is no keyboard that is clicky, RGB backlight, full size, you'd rather only exchange the thermostat. You are detail-oriented and want to make sure everything is addressed in one go."
    for i in range(max_steps):
        if i == 0:
            user_input = user.reset(instruction)
            cprint(f"(Step {i}) User: {user_input}", "grey")
        else:
            user_input = user.step(user_input)
            cprint(f"(Step {i}) User: {user_input}", "grey")

        # pass user input to agent
        agent_response = agent.step(user_input)
        cprint(f"(Step {i}) Agent: {agent_response}", "grey")


if __name__ == "__main__":
    fire.Fire(main)
