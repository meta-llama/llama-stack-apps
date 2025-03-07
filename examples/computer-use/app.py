#!/usr/bin/env python
import os
import time
import uuid

import fire
import pyautogui

# Import llama-stack client components
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.inference.utils import MessageAttachment
from utils import get_element_center, run_parser

# Global variable to store the latest OmniParser output
latest_omni_output = {}


# ------------------ Tool Implementations ------------------
@client_tool
def left_click_tool(box_id: int = None) -> str:
    """
    Performs a left click.
    If a box_id is provided, this tool looks up the coordinate from the global latest OmniParser output
    and clicks at the center of the identified element. Otherwise, it performs a left click at the current mouse position.

    text
    :param box_id: The ID of the element to click (optional).
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords = get_element_center(box_id)
        if coords:
            x, y = coords
            pyautogui.click(x, y)
            return f"Left-clicked on element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        pyautogui.click()
        return "Left-clicked at the current mouse position."


@client_tool
def right_click_tool(box_id: int = None) -> str:
    """
    Given a box_id, this tool looks up the coordinate from the global latest OmniParser output
    and simulates a right click.

    text
    :param box_id: The ID of the element to click.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords = get_element_center(box_id)
        if coords:
            x, y = coords
            pyautogui.rightClick(x, y)
            return f"Right-clicked on element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        pyautogui.rightClick()
        return "Right-clicked at the current mouse position."


@client_tool
def double_click_tool(box_id: int = None) -> str:
    """
    Performs a double click.
    If a box_id is provided, looks up the coordinate from the global latest OmniParser output and double-clicks at the element's center.
    Otherwise, double-clicks at the current mouse position.

    text
    :param box_id: The ID of the element to double-click (optional).
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords = get_element_center(box_id)
        if coords:
            x, y = coords
            pyautogui.doubleClick(x, y)
            return f"Double-clicked on element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        pyautogui.doubleClick()
        return "Double-clicked at the current mouse position."


@client_tool
def hover_tool(box_id: int) -> str:
    """
    Moves the mouse pointer to the center of an element identified by box_id to simulate hovering.

    text
    :param box_id: The ID of the element to hover over.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    coords = get_element_center(box_id)
    if coords:
        x, y = coords
        pyautogui.moveTo(x, y, duration=0.5)
        return f"Hovered over element with Box ID {box_id} at coordinates ({x}, {y})."
    return f"Box id {box_id} not found."


@client_tool
def mouse_move_tool(box_id: int = None) -> str:
    """
    Moves the mouse pointer to the given coordinates or to the center of an element if box_id is provided.

    text
    :param box_id: The ID of the element to move to.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords = get_element_center(box_id)
        if coords:
            x, y = coords
            pyautogui.moveTo(x, y, duration=0.5)
            return f"Moved mouse to element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        return "Error: No coordinates or Box ID provided."


@client_tool
def key_tool(key_name: str) -> str:
    """
    Simulates a key press.

    text
    :param key_name: The key to press.
    :returns: A message describing the action taken.
    """
    if key_name:
        pyautogui.press(key_name)
        return f"Pressed the key: '{key_name}'."
    return "Error: No key specified."


@client_tool
def scroll_up_tool() -> str:
    """
    Scrolls up on the screen.

    text
    :returns: A message describing the action taken.
    """
    pyautogui.scroll(20)
    return "Scrolled up."


@client_tool
def scroll_down_tool() -> str:
    """
    Scrolls down on the screen.

    text
    :returns: A message describing the action taken.
    """
    pyautogui.scroll(-20)
    return "Scrolled down."


@client_tool
def wait_tool(seconds: int = 2) -> str:
    """
    Pauses execution for a specified number of seconds.

    :param seconds: The number of seconds to wait.
    :returns: A message describing the action taken.
    """
    time.sleep(seconds)
    return f"Waited for {seconds} seconds."


@client_tool
def type_tool(box_id: str, words: str):
    """
    Given a box_id and a words string, this tool finds the coordinates from the OmniParser output,
    clicks to focus the corresponding UI element, and types the provided text.
    :param boxid: The ID of the element to type into.
    :param words: The text to type.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords = get_element_center(box_id)
        if coords:
            x, y = coords
            # Click to focus the UI element before typing.
            pyautogui.click(x, y)
            pyautogui.typewrite(words, interval=0.05)
            return f"Moved mouse to element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        return "No OmniParser output available."


@client_tool
def terminal_tool():
    """
    Stub tool for launching or switching to a terminal.
    """
    print("Simulating terminal launch... (stub)")
    # You might use os.system or other means to open/create a new terminal window.
    return "Terminal launched (stub)."


@client_tool
def homepage_tool():
    """
    Stub tool to return to the homepage state.
    """
    print("Returning to homepage... (stub)")
    return "Homepage restored (stub)."


# ------------------ Main Browser Agent Loop ------------------
class ComputerUseAgent:
    def __init__(self, max_step) -> None:
        self.action_history = []
        self.max_steps = max_step
        self.step = 0
        self.cur_output = None
        self.tools = [
            left_click_tool,
            right_click_tool,
            double_click_tool,
            hover_tool,
            mouse_move_tool,
            key_tool,
            scroll_up_tool,
            scroll_down_tool,
            wait_tool,
            type_tool,
            terminal_tool,
            homepage_tool,
        ]

    def run_agent(self, host: str = "localhost", port: int = 8321):
        """
        Sets up the Llama-Stack client and ReAct agent, gets a user query,
        and then iteratively uses the OmniParser server output and agent’s decision to drive UI actions.

        Workflow:
        1. Screenshot the current UI.
        2. Feed screenshot to OmniParser server via run_parser().
        3. Compose a conversation prompt with user query and the OmniParser output.
        4. Get a command from the ReAct agent (e.g., "click Box 1" or "type Box 2: clone llama-stack repo").
        5. Parse and execute the tool call.
        6. Repeat until the goal is reached or the maximum number of steps is executed.
        """
        # Initialize Llama-Stack client and ReAct agent
        client = LlamaStackClient(
            base_url=f"http://{host}:{port}",
            provider_data={"tavily_search_api_key": os.getenv("TAVILY_SEARCH_API_KEY")},
        )
        # Replace with the model you plan to use.
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct"

        # Register our browser action tools
        agent = ReActAgent(
            client=client,
            model=model,
            client_tools=self.tools,
            json_response_format=True,
        )

        session_id = agent.create_session(f"browser-agent-session-{uuid.uuid4().hex}")
        user_query = input("Enter your query (e.g., 'clone llama-stack repo'): ")
        print("Starting browser agent for query:", user_query)

        while self.step < self.max_steps:
            print(f"\n---- Step {self.step + 1} ----")
            # Take a screenshot of the current screen.
            screenshot_path = f"temp_screenshot_{self.step}.png"
            screen_image = pyautogui.screenshot()
            screen_image.save(screenshot_path)
            print("Screenshot taken and saved as:", screenshot_path)

            # Call OmniParser with the screenshot to get structured UI info.
            global latest_omni_output
            labled_img, label_coordinates, parsed_content_list = run_parser(
                screenshot_path
            )
            latest_omni_output = parsed_content_list
            print("OmniParser output:", latest_omni_output)
            parsed_content_list = "\n".join(
                [f"icon {i}: " + str(v) for i, v in enumerate(parsed_content_list)]
            )
            self.cur_output = parsed_content_list
            # Prepare the prompt for the agent that includes the user query and current UI state.
            omni_info = f"OmniParser Output: {latest_omni_output}"

            prompt_instruction = (
                f"User query: {user_query}\n"
                f"Current Screen Info: {omni_info}\n"
                "You should look at the info above and determine next action."
                "Look at the tool availble to you and respond with a tool-call command like 'action': {'tool_name': 'right_click_tool', 'tool_params': {'box_id': 22} or 'action': {'tool_name': 'type_tool', 'tool_params': {'box_id': 12, words: SOME_TEXT } "
                "If the goal is reached, you should just output response like:{'thought': 'Now I reached goal and complete the task.','action': null, 'answer': 'Task completed, No more action needed'}."
            )
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {"url": {"uri": MessageAttachment.base64(labled_img)}},
                    },
                    {
                        "type": "text",
                        "text": prompt_instruction,
                    },
                ],
            }

            # Send the turn to the agent; stream output if desired.
            response = agent.create_turn(
                messages=[message],
                session_id=session_id,
                stream=True,
            )

            final_command = ""
            current_response = ""
            for log in EventLogger().log(response):
                if hasattr(log, "content"):
                    # print(f"Debug Response: {log.content}")
                    if "tool_execution>" in str(log):
                        current_response += (
                            " <tool-begin> " + log.content + " <tool-end> "
                        )
                    else:
                        final_command += log.content
                        current_response += log.content

            # Check if the agent has signaled that the goal is reached.
            if "Goal reached" in final_command:
                print("Goal reached. Ending session.")
                break

            # Optional: wait a short duration for UI to update after the tool call.
            time.sleep(2)
            self.step += 1


if __name__ == "__main__":
    agent = ComputerUseAgent(max_step=10)

    fire.Fire(agent.run_agent())
