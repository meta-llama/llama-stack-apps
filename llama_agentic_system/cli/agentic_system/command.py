# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import textwrap

from llama_toolchain.cli.subcommand import Subcommand

from llama_agentic_system.cli.agentic_system.configure import AgenticSystemConfigure


class AgenticSystemParser(Subcommand):
    """Llama cli for agentic system apis"""

    def __init__(self, subparsers: argparse._SubParsersAction):
        super().__init__()
        self.parser = subparsers.add_parser(
            "agentic_system",
            prog="llama agentic_system",
            description="Run agentic_system comamnds",
            epilog=textwrap.dedent(
                """
                Example:
                    llama agentic_system <sub-command> <options>
                """
            ),
        )

        subparsers = self.parser.add_subparsers(title="agentic_system_subcommands")

        # Add sub-commandsa
        AgenticSystemConfigure.create(subparsers)
