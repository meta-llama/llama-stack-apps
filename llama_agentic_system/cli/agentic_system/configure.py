# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import json
import os

from pathlib import Path

import yaml

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.common.config_dirs import LLAMA_STACK_CONFIG_DIR
from termcolor import cprint


class AgenticSystemConfigure(Subcommand):

    def __init__(self, subparsers: argparse._SubParsersAction):
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama agentic_system configure",
            description="Configure llama agentic system configs",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_agentic_system_configure_cmd)

    def _add_arguments(self):
        pass

    def _run_agentic_system_configure_cmd(self, args: argparse.Namespace) -> None:
        from llama_toolchain.common.prompt_for_config import prompt_for_config
        from llama_toolchain.common.serialize import EnumEncoder

        from llama_agentic_system.config import AgenticSystemConfig

        os.makedirs(LLAMA_STACK_CONFIG_DIR / "agentic_system", exist_ok=True)
        config_path = Path(LLAMA_STACK_CONFIG_DIR) / "agentic_system/config.yaml"

        existing_config = None
        if config_path.exists():
            cprint(
                "Configuration already exists. Will overwrite...",
                "yellow",
                attrs=["bold"],
            )
            with open(config_path, "r") as fp:
                existing_config = yaml.safe_load(fp)

        config = prompt_for_config(AgenticSystemConfig, existing_config)
        with open(config_path, "w") as fp:
            config = json.loads(json.dumps(config.dict(), cls=EnumEncoder))
            fp.write(yaml.dump(config, sort_keys=False))
