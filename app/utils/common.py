# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import asyncio
import os
import sys

import mesop as me

from .chat import State

if sys.version_info < (3, 10):
    raise Exception("Demo requires Python 3.10+")
sys.path += [f'{os.path.expanduser("~/llama-agent-system")}']

UPLOADS_DIR = "app/uploads/"
CHUNK_SIZE = 1024
DISTRIBUTION_PORT = os.environ.get("DISTRIBUTION_PORT", 5000)
DISTRIBUTION_HOST = os.environ.get("DISTRIBUTION_HOST", "localhost")
DISABLE_SAFETY = bool(int(os.environ.get("DISABLE_SAFETY", "0")))


def sync_generator(loop, async_generator):
    asyncio.set_event_loop(loop)

    def generator():
        while True:
            try:
                yield loop.run_until_complete(async_generator.__anext__())
            except StopAsyncIteration:
                break

    return generator()


def on_attach(e: me.UploadEvent):
    path = os.path.join(UPLOADS_DIR, e.file.name)
    state = me.state(State)

    if not os.path.isdir(UPLOADS_DIR):
        os.mkdir(UPLOADS_DIR)

    with open(path, "wb") as f:
        f.write(e.file.read())
        state.pending_attachment_path = path
        state.pending_attachment_mime_type = e.file.mime_type
