# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import base64
import os
import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Sequence, Union

import mesop as me

from mesop.components.uploader.uploader import UploadEvent

from llama_toolchain.inference.api import *  # noqa: F403
from llama_toolchain.agentic_system.api import *  # noqa: F403
from llama_toolchain.safety.api import *  # noqa: F403

MAX_VIOLATIONS = 3


STDOUT_CUTOFF_LENGTH = 200

_MESSAGE_TYPE_TEXT = "text"
_MESSAGE_TYPE_IMAGE = "image"

_SPECIAL_PYTHON_TOKEN = "<|python_tag|>"
_SPECIAL_STDOUT_DELIMITER = "[stdout]"

_COLOR_BACKGROUND = "#f0f4f8"
_COLOR_CHAT_BUBBLE_YOU = "#f2f2f2"
_COLOR_CHAT_BUBBLE_BOT = "#e8f3ff"
_COLOR_CHAT_BUBBLE_TOOL = "#d0dae5"
_COLOR_BUTTON = "#1d65c1"

_DEFAULT_PADDING = me.Padding.all(20)
_DEFAULT_BORDER_SIDE = me.BorderSide(width="1px", style="solid", color="#ececec")

_LABEL_BUTTON = "send"
_LABEL_BUTTON_IN_PROGRESS = "pending"
_LABEL_INPUT = "Enter your prompt"

_STYLE_APP_CONTAINER = me.Style(
    background=_COLOR_BACKGROUND,
    display="grid",
    height="100vh",
    grid_template_columns="repeat(1, 1fr)",
)
_STYLE_TITLE = me.Style(padding=me.Padding(left=5))

_STYLE_CHAT_UI_CONTAINER = me.Style(
    display="flex",
    flex_direction="column",
    margin=me.Margin.symmetric(vertical=0, horizontal="auto"),
    width="min(1024px, 100%)",
    height="100vh",
    background="#fff",
    box_shadow=("0 3px 1px -2px #0003, 0 2px 2px #00000024, 0 1px 5px #0000001f"),
    padding=me.Padding(top=20, left=20, right=20, bottom=20),
)

_STYLE_CHAT_BOX = me.Style(
    height="100%",
    overflow_y="scroll",
    flex_grow=1,
    padding=_DEFAULT_PADDING,
    border_radius="10px",
    border=me.Border(
        left=_DEFAULT_BORDER_SIDE,
        right=_DEFAULT_BORDER_SIDE,
        top=_DEFAULT_BORDER_SIDE,
        bottom=_DEFAULT_BORDER_SIDE,
    ),
)
_STYLE_CHAT_INPUT = me.Style(width="100%")
_STYLE_CHAT_INPUT_BOX = me.Style(
    padding=me.Padding(top=30), display="flex", flex_direction="row"
)
_STYLE_CHAT_BUTTON = me.Style(
    margin=me.Margin(top=8, left=8),
    background=_COLOR_BUTTON,
)
_STYLE_CHAT_BUBBLE_PLAINTEXT = me.Style(margin=me.Margin.symmetric(vertical=15))


@dataclass
class StepStatus:
    step_type: StepType
    content: Union[InterleavedTextMedia, ShieldResponse]
    show_tool_response: bool = False


RenderableOutputType = Union[Message, StepStatus]

# HACK: Mesop's state management has issues with serializing more complex types
# like Union (message content) and PIL_Image. To avoid duplicative message types,
# we'll use a global dict to store the messages and just store message keys in state.
KEY_TO_OUTPUTS: Dict[str, RenderableOutputType] = {}


@me.stateclass
class State:
    input: str
    output: List[str]
    in_progress: bool = False
    pending_attachment_path: str
    pending_attachment_mime_type: str
    violation_count: int = 0
    debug_mode: bool = False


def _make_style_chat_bubble_wrapper(role: str) -> me.Style:
    """Generates styles for chat bubble position.

    Args:
      role: Chat bubble alignment depends on the role
    """
    align_items = "end" if role == Role.user.value else "start"
    return me.Style(
        display="flex",
        flex_direction="column",
        align_items=align_items,
        justify_content="center",
    )


def is_tool_op(op: RenderableOutputType) -> bool:
    return isinstance(op, StepStatus) and op.step_type in (
        StepType.inference,
        StepType.tool_execution,
        StepType.shield_call,
    )


def _make_chat_bubble_style(op: RenderableOutputType, role: Role) -> me.Style:
    if is_tool_op(op):
        return me.Style(
            font_size="12px",
            font_family="monospace",
            width="80%",
            background=_COLOR_CHAT_BUBBLE_TOOL,
            border_radius="15px",
            padding=me.Padding(top=10, bottom=10, left=10, right=10),
            align_items="center",
            justify_content="center",
            margin=me.Margin(bottom=8),
            border=me.Border(
                left=_DEFAULT_BORDER_SIDE,
                right=_DEFAULT_BORDER_SIDE,
                top=_DEFAULT_BORDER_SIDE,
                bottom=_DEFAULT_BORDER_SIDE,
            ),
        )

    background = (
        _COLOR_CHAT_BUBBLE_YOU if role == Role.user.value else _COLOR_CHAT_BUBBLE_BOT
    )

    return me.Style(
        width="80%" if isinstance(op.content, str) else None,
        font_size="16px",
        background=background,
        border_radius="15px",
        padding=me.Padding(top=6, bottom=6, left=10, right=10),
        align_items="center",
        justify_content="center",
        margin=me.Margin(bottom=8),
        border=me.Border(
            left=_DEFAULT_BORDER_SIDE,
            right=_DEFAULT_BORDER_SIDE,
            top=_DEFAULT_BORDER_SIDE,
            bottom=_DEFAULT_BORDER_SIDE,
        ),
    )


def on_blur(e: me.InputBlurEvent):
    state = me.state(State)
    state.input = e.value


def render(content: Union[str, Attachment, Sequence[Union[str, Attachment]]]):
    if isinstance(content, str):
        return me.markdown(content.replace("\n", "\n\n"))
    elif isinstance(content, Attachment) and "image" in content.mime_type:
        uri = content.url.uri
        if uri.startswith("file://"):
            filepath = uri[len("file://") :]
            with open(filepath, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
                image_url = f"data:image/png;base64,{image_base64}"
        else:
            image_url = uri

        return me.image(src=image_url, style=me.Style(height=300))
    elif isinstance(content, Attachment):
        with me.box(
            style=me.Style(
                display="flex", flex_direction="row", justify_content="center"
            )
        ):
            filename = content.url.uri.split("/")[-1]
            me.icon(icon="attachment", style=me.Style(margin=me.Margin(right=4)))
            me.text(f"{filename}\n", style=me.Style(font_family="monospace"))
    elif isinstance(content, list) or isinstance(content, tuple):
        return [render(subcontent) for subcontent in content]


def chat(
    transform: Callable[[str, List[Message]], Generator[str, None, None] | str],
    *,
    title: str,
    bot_user: str,
    on_attach: Callable[[UploadEvent], Any],
    moderated: bool = False,
):
    state = me.state(State)

    state.moderated = moderated

    def on_click_submit(e: me.ClickEvent):
        yield from submit()

    def on_input_enter(e: me.InputEnterEvent):
        state = me.state(State)
        state.input = e.value
        yield from submit()

    def submit():
        state = me.state(State)
        if state.in_progress or not state.input:
            return
        state.in_progress = True

        input = state.input
        state.input = ""

        output = state.output
        if output is None:
            output = []

        # Parse any pending attachment into the newly sent message and clear it.
        content = (
            [
                input,
                Attachment(
                    url=URL(
                        uri=f"file://{os.path.abspath(state.pending_attachment_path)}"
                    ),
                    mime_type=state.pending_attachment_mime_type,
                ),
            ]
            if state.pending_attachment_path is not None
            and state.pending_attachment_path != ""
            else input
        )
        state.pending_attachment_path = None
        state.pending_attachment_mime_type = None

        msg_uuid = str(uuid.uuid4())
        KEY_TO_OUTPUTS[msg_uuid] = UserMessage(content=content)
        output.append(msg_uuid)
        state.output = output

        me.scroll_into_view(key="scroll-to")
        yield

        cur_uuids = set(state.output)
        for op_uuid, op in transform(content):
            KEY_TO_OUTPUTS[op_uuid] = op
            if op_uuid not in cur_uuids:
                output.append(op_uuid)
                cur_uuids.add(op_uuid)

            state.output = output
            yield

        state.in_progress = False
        yield

    def on_debug_mode_change(event: me.SlideToggleChangeEvent):
        s = me.state(State)
        s.debug_mode = not s.debug_mode

    def codeblock(content: str):
        lines = content.split("\n")
        with me.box(
            style=me.Style(
                display="flex",
                flex_direction="column",
                margin=me.Margin(top=6, bottom=6, left=6, right=6),
            )
        ):
            for line in lines:
                me.text(line)

    def on_click_toggle_tool_response(e: me.ClickEvent):
        op = KEY_TO_OUTPUTS[e.key]
        op.show_tool_response = not op.show_tool_response

    def render_tool(op_uuid: str, op: RenderableOutputType):
        if isinstance(op, StepStatus) and op.step_type == StepType.inference:
            me.text("Executing tool...", type="subtitle-2")
            with me.box(style=_make_chat_bubble_style(op, Role.assistant.value)):
                return codeblock(op.content)

        if isinstance(op.content, list):
            with me.box(style=_make_chat_bubble_style(op, Role.assistant.value)):
                return render(op.content)

        elif isinstance(op, StepStatus) and op.step_type == StepType.shield_call:
            if state.debug_mode:
                shield_response = op.content
                with me.box(
                    style=me.Style(
                        display="flex",
                        flex_direction="row",
                        justify_content="center",
                        align_items="center",
                        margin=me.Margin(top=8, bottom=8),
                    )
                ):
                    me.icon(
                        icon="health_and_safety",
                        style=me.Style(margin=me.Margin(right=4)),
                    )
                    me.text(
                        f"Violation type: {shield_response.violation_type} ({state.violation_count} violations so far)"
                    )
        elif isinstance(op, StepStatus) and op.step_type == StepType.tool_execution:
            if isinstance(op.content, str) and _SPECIAL_STDOUT_DELIMITER in op.content:
                stdout_match = re.search(
                    r"\[stdout\](.*?)\[/stdout\]", op.content, re.DOTALL
                )
                if not stdout_match:
                    raise Exception(f"Unable to parse stdout: {op.content}")
                stdout = stdout_match.group(1)
                with me.box(style=_make_chat_bubble_style(op, Role.assistant.value)):
                    if len(stdout) > STDOUT_CUTOFF_LENGTH:
                        codeblock(stdout[:STDOUT_CUTOFF_LENGTH])
                        me.text(
                            f"(remaining {len(stdout) - STDOUT_CUTOFF_LENGTH} characters omitted)",
                            type="subtitle-2",
                        )
                    else:
                        return codeblock(stdout)
            else:
                # TODO: Hack to disable photogen showing tool response due to rendering
                if str(BuiltinTool.photogen) not in op.content:
                    with me.box(style=_make_chat_bubble_style(op, Role.ipython.value)):
                        if op.show_tool_response:
                            me.button(
                                "Hide tool response △",
                                on_click=on_click_toggle_tool_response,
                                key=op_uuid,
                            )
                            codeblock(str(op.content))
                        else:
                            me.button(
                                "Show tool response ▽",
                                on_click=on_click_toggle_tool_response,
                                key=op_uuid,
                            )

    with me.box(style=_STYLE_APP_CONTAINER):
        with me.box(style=_STYLE_CHAT_UI_CONTAINER):
            if state.moderated and state.violation_count >= MAX_VIOLATIONS:
                me.text("Chat Closed", type="headline-5", style=_STYLE_TITLE)
            elif title:
                me.text(title, type="headline-5", style=_STYLE_TITLE)

            with me.box(style=_STYLE_CHAT_BOX):
                for op_uuid in state.output:
                    try:
                        op = KEY_TO_OUTPUTS[op_uuid]
                    except KeyError:
                        print(f"Missing key {op_uuid} in KEY_TO_OUTPUTS")
                        continue

                    role = op.role if hasattr(op, "role") else Role.assistant.value
                    with me.box(style=_make_style_chat_bubble_wrapper(role)):
                        if is_tool_op(op):
                            render_tool(op_uuid, op)
                        else:
                            with me.box(style=_make_chat_bubble_style(op, role)):
                                render(op.content)

                if state.in_progress:
                    with me.box(key="scroll-to", style=me.Style(height=300)):
                        pass
            if state.moderated and state.violation_count >= MAX_VIOLATIONS:
                return
            with me.box(style=_STYLE_CHAT_INPUT_BOX):
                with me.box(style=me.Style(flex_grow=1)):
                    me.input(
                        label=_LABEL_INPUT,
                        disabled=state.in_progress,
                        key=f"{len(state.output)}",
                        on_blur=on_blur,
                        on_enter=on_input_enter,
                        style=_STYLE_CHAT_INPUT,
                    )
                with me.content_button(
                    color="primary",
                    type="flat",
                    disabled=state.in_progress,
                    on_click=on_click_submit,
                    style=_STYLE_CHAT_BUTTON,
                ):
                    me.icon(
                        _LABEL_BUTTON_IN_PROGRESS
                        if state.in_progress
                        else _LABEL_BUTTON
                    )

            with me.box(
                style=me.Style(
                    display="flex",
                    flex_direction="row",
                    justify_content="space-between",
                )
            ):
                me.uploader(
                    label="Upload",
                    on_upload=on_attach,
                    key=f"{len(state.output)}",
                    style=me.Style(
                        color=_COLOR_BUTTON,
                    ),
                )

                me.slide_toggle(label="Debug Mode", on_change=on_debug_mode_change)
