# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from enum import Enum
from typing import List, Literal, Optional, Union

from hydra.core.config_store import ConfigStore

from hydra_zen import builds

from llama_toolchain.inference.api.config import InferenceConfig
from llama_toolchain.safety.api.config import SafetyConfig

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class ImplType(Enum):
    inline = "inline"
    remote = "remote"


class InlineImplConfig(BaseModel):
    impl_type: Literal[ImplType.inline.value] = ImplType.inline.value
    inference_config: InferenceConfig


class RemoteImplConfig(BaseModel):
    impl_type: Literal[ImplType.remote.value] = ImplType.remote.value
    url: str = Field(..., description="The URL of the remote module")


# TODO: Move to toolchain/safety/config.py
class LlamaGuardShieldConfig(BaseModel):
    model_dir: str
    excluded_categories: List[str]
    disable_input_check: bool = False
    disable_output_check: bool = False


class PromptGuardShieldConfig(BaseModel):
    model_dir: str


class AgenticSystemConfig(BaseModel):
    impl_config: Annotated[
        Union[InlineImplConfig, RemoteImplConfig],
        Field(discriminator="impl_type"),
    ]
    safety_config: Optional[SafetyConfig] = None


AgenticSystemHydraConfig = builds(AgenticSystemConfig)

cs = ConfigStore.instance()
cs.store(name="agentic_system_config", node=AgenticSystemHydraConfig)
