# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

# Run from top level dir as:
# PYTHONPATH=. python3 scripts/run_e2e_tests.py

import asyncio
import os
import signal
import subprocess
import time

import fire

from llama_toolchain.agentic_system.utils import (
    get_agent_with_custom_tools,
    make_agent_config_with_custom_tools,
)


class CleanChildProcesses:
    def __enter__(self):
        os.setpgrp()  # create new process group, become its leader

    def __exit__(self, type, value, traceback):
        try:
            os.killpg(0, signal.SIGINT)  # kill all processes in my group
        except KeyboardInterrupt:
            # SIGINT is delievered to this process as well as the child processes.
            # Ignore it so that the existing exception, if any, is returned. This
            # leaves us with a clean exit code if there was no exception.
            pass


async def run_main(host: str, port: int):
    with CleanChildProcesses():
        print("Starting server...")
        server_process = subprocess.Popen(
            [
                "python3",
                "agentic_system/impl/server.py",
                "--port",
                str(port),
            ],
            # Hide from printing output to make it easier to read test output
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        while True:
            try:
                agent_config = await make_agent_config_with_custom_tools()
                agent = await get_agent_with_custom_tools(
                    host=host,
                    port=port,
                    agent_config=agent_config,
                    custom_tools=[],
                )
                break
            except Exception:
                print("Waiting for server to be ready...")
                time.sleep(5)

        print("Running tests...")
        # TODO: Pass in host and port into tests
        test_process = subprocess.Popen(
            [
                "python3",
                "tests/test_e2e.py",
            ],
        )
        test_process.wait()


def main(host: str = "localhost", port: int = 5000):
    asyncio.run(run_main(host=host, port=port))


if __name__ == "__main__":
    fire.Fire(main)
