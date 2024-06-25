# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import signal

import fire

from llama_agentic_system.api import *  # noqa: F401 F403
from dotenv import load_dotenv

from fastapi import FastAPI

from hydra_zen import instantiate

from llama_toolchain.safety.shields import LlamaGuardShield, PromptGuardShield
from llama_toolchain.utils import parse_config

from llama_agentic_system.api_instance import get_agentic_system_api_instance
from llama_agentic_system.utils import get_config_dir


load_dotenv()

GLOBAL_CONFIG = None
AgenticSystemApiInstance = None


def get_config():
    return GLOBAL_CONFIG


def handle_sigint(*args, **kwargs):
    print("SIGINT or CTRL-C detected. Exiting gracefully", args)
    loop = asyncio.get_event_loop()
    for task in asyncio.all_tasks(loop):
        task.cancel()
    loop.stop()


app = FastAPI()


@app.on_event("startup")
async def startup():
    global AgenticSystemApiInstance

    config = get_config()

    agentic_system_config = instantiate(config["agentic_system_config"])
    AgenticSystemApiInstance = await get_agentic_system_api_instance(
        agentic_system_config
    )

    # Manage Safety Shields
    if agentic_system_config.safety_config is not None:
        safety_config = agentic_system_config.safety_config
        assert (
            safety_config is not None
        ), "safety_config must be provided, else set disable_safety to True"

        # Check for LlamaGuard Shield
        shield_cfg = safety_config.llama_guard_shield
        if shield_cfg is not None:
            _ = LlamaGuardShield.instance(
                model_dir=shield_cfg.model_dir,
                excluded_categories=shield_cfg.excluded_categories,
                disable_input_check=shield_cfg.disable_input_check,
                disable_output_check=shield_cfg.disable_output_check,
            )

        # Check for PromptGuard Shield
        shield_cfg = safety_config.prompt_guard_shield
        if shield_cfg is not None:
            _ = PromptGuardShield.instance(shield_cfg.model_dir)


@app.post(
    "/agentic_system/create",
    response_model=AgenticSystemCreateResponse,
)
async def agentic_system_create(
    exec_request: AgenticSystemCreateRequest,
):
    return await AgenticSystemApiInstance.create_agentic_system(exec_request)


@app.post(
    "/agentic_system/session/create",
    response_model=AgenticSystemSessionCreateResponse,
)
async def create_agentic_system_session(
    request: AgenticSystemSessionCreateRequest,
) -> AgenticSystemSessionCreateResponse:
    return await AgenticSystemApiInstance.create_agentic_system_session(request)


@app.post(
    "/agentic_system/turn/create",
    response_model=None,
)
async def create_agentic_system_turn(request: AgenticSystemTurnCreateRequest):
    async def sse_generator(event_gen):
        try:
            async for event in event_gen:
                yield f"data: {event.json()}\n\n"
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            print("Generator cancelled")
            await event_gen.aclose()

    async def event_gen():
        async for event in AgenticSystemApiInstance.create_agentic_system_turn(request):
            yield event

    return StreamingResponse(
        sse_generator(event_gen()),
        media_type="text/event-stream",
    )


def main(config_path: str = "inline", port: int = 5001, disable_ipv6: bool = False):
    global GLOBAL_CONFIG
    config_dir = get_config_dir()

    GLOBAL_CONFIG = parse_config(config_dir, config_path)

    signal.signal(signal.SIGINT, handle_sigint)
    import uvicorn

    # FYI this does not do hot-reloads
    listen_host = "::" if not disable_ipv6 else "0.0.0.0"
    print(f"Listening on {listen_host}:{port}")
    uvicorn.run(app, host=listen_host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
