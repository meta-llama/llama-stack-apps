# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from .envs.retail.data import load_data as load_retail_data


class BaseEnv:
    def __init__(self, env_name: str):
        if env_name == "retail":
            self.env_name = env_name
            self.data = load_retail_data()
        else:
            raise ValueError(f"Invalid environment name: {env_name}")

    def reset(self):
        self.data = load_retail_data()

    def __eq__(self, other):
        return self.data == other.data
