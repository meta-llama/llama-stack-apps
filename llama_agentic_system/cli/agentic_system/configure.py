# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import argparse
import os
import textwrap

from pathlib import Path

import pkg_resources

from llama_toolchain.cli.subcommand import Subcommand
from llama_toolchain.utils import DEFAULT_DUMP_DIR

from llama_agentic_system.utils import get_config_dir


class AgenticSystemConfigure(Subcommand):

    def __init__(self, subparsers: argparse._SubParsersAction):
        self.parser = subparsers.add_parser(
            "configure",
            prog="llama agentic_system configure",
            description="Configure llama agentic system configs",
            epilog=textwrap.dedent(
                """
                Example:
                    llama agentic_system configure
                """
            ),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        self._add_arguments()
        self.parser.set_defaults(func=self._run_agentic_system_configure_cmd)

    def _add_arguments(self):
        pass

    def input_with_defaults(self, prompt, default_value):
        res = input(f"{prompt} [Default: {default_value}] (hit enter to use default): ")
        if not res:
            print(f"Using default value: {default_value}")
            return default_value

        return res

    def read_checkpoint_dir(self, name: str) -> str:
        checkpoint_dir = os.path.expanduser(
            input(f"Enter the checkpoint dir for {name}: ")
        )
        checkpoint_dir = os.path.expanduser(checkpoint_dir.strip())
        assert os.path.isdir(
            checkpoint_dir
        ), f"{checkpoint_dir} is not a valid directory"

        return checkpoint_dir

    def read_user_inputs(self):
        inference_server_url = self.input_with_defaults(
            "Enter the inference server endpoint", "http://localhost:5000"
        )
        default_llama_guard_model_dir = os.path.join(
            DEFAULT_DUMP_DIR,
            "checkpoints",
        )
        llama_guard_model_dir = self.read_checkpoint_dir("llama_guard")
        prompt_guard_model_dir = self.read_checkpoint_dir("prompt_guard")

        return inference_server_url, llama_guard_model_dir, prompt_guard_model_dir

    def write_output_yaml(
        self,
        inference_server_url,
        llama_guard_model_dir,
        prompt_guard_model_dir,
        yaml_output_path,
    ):
        default_conf_path = pkg_resources.resource_filename(
            "llama_agentic_system", "data/default.yaml"
        )
        with open(default_conf_path, "r") as f:
            yaml_content = f.read()

        yaml_content = yaml_content.format(
            inference_server_url=inference_server_url,
            llama_guard_model_dir=llama_guard_model_dir,
            prompt_guard_model_dir=prompt_guard_model_dir,
        )

        with open(yaml_output_path, "w") as yaml_file:
            yaml_file.write(yaml_content.strip())

        print(f"YAML configuration has been written to {yaml_output_path}")

    def _run_agentic_system_configure_cmd(self, args: argparse.Namespace) -> None:
        inference_server_url, llama_guard_model_dir, prompt_guard_model_dir = (
            self.read_user_inputs()
        )
        print(inference_server_url, llama_guard_model_dir, prompt_guard_model_dir)

        base_dir = get_config_dir()
        os.makedirs(base_dir, exist_ok=True)
        yaml_output_path = Path(base_dir) / "inline.yaml"

        self.write_output_yaml(
            inference_server_url,
            llama_guard_model_dir,
            prompt_guard_model_dir,
            yaml_output_path,
        )
