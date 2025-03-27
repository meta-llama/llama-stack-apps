#!/usr/bin/env python
import base64
import os
import time
import uuid
import json
import pyautogui
from llama_stack.distribution.library_client import LlamaStackAsLibraryClient
from pydantic import BaseModel
from rich.pretty import pprint
# Import llama-stack client components
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.agent import Agent

from llama_stack_client.lib.agents.react.agent import ReActAgent
from llama_stack_client.lib.inference.utils import MessageAttachment
from utils import (
    convert_to_base64,
    get_element_center,
    parse_omni_parser_output,
    run_parser,
)
class AnswerFormat(BaseModel):
    thoughts: str
    action: str
    answer: str
# Global variable to store the latest OmniParser output
latest_omni_output = {}


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        base64_url = f"data:image/png;base64,{base64_string}"
        return base64_url

def encode_image_str(image_path):
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string
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
        coords =get_element_center(box_id,latest_omni_output)
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
        coords =get_element_center(box_id,latest_omni_output)
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
def double_click_tool(box_id: int) -> str:
    """
    Performs a double click.
    If a box_id is provided, looks up the coordinate from the global latest OmniParser output and double-clicks at the element's center.
    Otherwise, double-clicks at the current mouse position.

    text
    :param box_id: The ID of the element to double-click.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords =get_element_center(box_id,latest_omni_output)
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
    coords =get_element_center(box_id,latest_omni_output)
    if coords:
        x, y = coords
        pyautogui.moveTo(x, y, duration=0.5)
        return f"Hovered over element with Box ID {box_id} at coordinates ({x}, {y})."
    return f"Box id {box_id} not found."


@client_tool
def mouse_move_tool(box_id: int) -> str:
    """
    Moves the mouse pointer to the given coordinates or to the center of an element if box_id is provided.

    text
    :param box_id: The ID of the element to move to.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords =get_element_center(box_id,latest_omni_output)
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
def scroll_up_tool(num_pixel: int) -> str:
    """
    Scrolls up on the screen.

    text
    :param num_pixel: The number of pixels to scroll.
    :returns: A message describing the action taken.
    """
    pyautogui.scroll(num_pixel)
    return f"Scrolled up for {num_pixel} pixel."


@client_tool
def scroll_down_tool(num_pixel: int) -> str:
    """
    Scrolls down on the screen.
    :param num_pixel: The number of pixels to scroll.
    text
    :returns: A message describing the action taken.
    """
    pyautogui.scroll(-20)
    return f"Scrolled down for {num_pixel} pixel."


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
def output_final_answer(answer: str) -> str:
    """
    Outputs the answer to the user and stops the agent. Use this when the task is completed.

    :param answer: The answer to output.
    :returns: A message describing the action taken.
    """
    print(f"Outputting answer: {answer}")
    exit()
    return f"Outputting answer: {answer}"
@client_tool
def type_tool(box_id: int, words: str, enter: bool = False) -> str:
    """
    Given a box_id and a words string, this tool finds the coordinates from the OmniParser output,
    clicks to focus the corresponding UI element, and types the provided text.
    :param box_id: The ID of the element to type into.
    :param words: The text to type.
    :param enter: Whether to press the Enter key after typing so that words will be entered.
    :returns: A message describing the action taken.
    """
    global latest_omni_output
    if box_id is not None:
        coords =get_element_center(box_id,latest_omni_output)
        if coords:
            x, y = coords
            # Click to focus the UI element before typing.
            pyautogui.click(x, y)
            pyautogui.typewrite(words, interval=0.05)
            if enter:
                pyautogui.press("enter")
            return f"Moved mouse to element with Box ID {box_id} at coordinates ({x}, {y})."
        else:
            return f"Box id {box_id} not found."
    else:
        return "No OmniParser output available."


# ------------------ Main Computer agent Loop ------------------
class ComputerUseAgent:
    def __init__(self, max_step) -> None:
        self.action_history = []
        self.max_steps = max_step
        self.step = 0
        self.client = None
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
            output_final_answer
        ]

    def run_agent(self, host="localhost", port=8321):
        """
        Sets up the Llama-Stack client and ReAct agent, gets a user query,
        and then iteratively uses the OmniParser server output and agent’s decision to drive UI actions.

        Workflow:
        1. Screenshot the current UI.
        2. Feed screenshot to OmniParser server via run_parser().
        3. Compose a conversation prompt with user query and the OmniParser output.
        4. ReAct agent will use prompt + screenshot to do tool_call.
        5. Repeat until the goal is reached or the maximum number of steps is executed.
        """
        # Replace with the model you plan to use.
        model = "meta-llama/Llama-3.2-90B-Vision-Instruct"
       
        if host == "localhost":
            client = LlamaStackClient(
            base_url=f"http://{host}:{port}",
            )
        print('using model: ', model)
        agent = ReActAgent(
            client=client,
            model=model,
            client_tools=self.tools,
            #json_response_format=True,
            sampling_params={"top_p": 1.0, "temperature": 0.0},
        )
        

        user_query = "Open Safari, use Google to search 'current Meta stock price' and return that price"  #  Example query - will loop forever with this.
        # user_query = input("Enter your query (e.g., 'clone llama-stack repo'): ")
        print("Starting computer agent for query:", user_query)

        while self.step < self.max_steps:
            print(f"\n---- Step {self.step + 1} ----")
            # Create a new session for each step because the context is too long for the agent to remember.
            session_id = agent.create_session(f"computer-use-session-{uuid.uuid4().hex}-step-{self.step}")
            # Take a screenshot of the current screen.
            screenshot_path = f"/tmp/temp_screenshot_{self.step}.png"
            screen_image = pyautogui.screenshot()
            # Resize to 640x400 to save space and reduce latency in sending the image to the server.
            resized_image = screen_image.resize((640, 400))
            # Save the resized image
            resized_image.save(screenshot_path)
            #screen_image.save(screenshot_path)
            print("Screenshot taken and saved as:", screenshot_path)

            # Call OmniParser with the screenshot to get structured UI info.
            global latest_omni_output
            _, raw_parse_output = run_parser(screenshot_path)
            parsed_content_list = parse_omni_parser_output(raw_parse_output)
            latest_omni_output = parsed_content_list
            # Prepare the prompt for the agent that includes the user query and current UI state.
            screen_info = ""
            for content in parsed_content_list:
                screen_info += f"Box ID {content['id']}, type: {content['type']}, interactivity: {content['interactivity']},content: {content['content']}\n"
            action_history = "\n".join([f'Step {step}: {history}' for step, history in enumerate(self.action_history)])
            prompt_instruction = f"""
            Task: {user_query}

            Current State:
            - Action History: {action_history}
            - Screen Elements: {len(screen_info.split('ID:'))-1} interactive elements detected

            Element Details:
            {screen_info}
            """
            prompt_instruction += """
            Analysis Requirements:
            1. First, analyze the current state of the screen and the action history
            2. Identify the most relevant elements that will help complete the task, usually the most recent action should be the most relevant
            3. Determine the optimal next action based on the task goal
            4. Select the appropriate tool and parameters for that action
            5. If the task appears complete, use output_final_answer

            Common Mistakes to Avoid:
            - Focus on the irrelevant elements, such as the background or other windows, the most recent action should be the most relevant
            - Clicking on elements that don't exist or are irrelevant
            - Typing in non-input fields, MUST USE the type_tool to type instead of clicking on the input field
            - Clicking on elements that are not clickable
            - Declaring task completion prematurely
            - Repeating the same action multiple times without progress
            - Ignoring previous failed attempts

            IMPORTANT: YOUR RESPONSE MUST BE IN VALID JSON FORMAT WITH DOUBLE QUOTES. Your response must follow this exact structure:

            {
            'thought': 'Your thought process analyzing the screenshot and determining the best action',
            'action': {
                'tool_name': 'left_click_tool',
                'tool_params': [
                {
                    'name': 'box_id',
                    'value': 22
                }
                ]
            },
            'answer': 'Description of the action taken, the name of the element () you interacted with, and the next step as detailed as possible'
            }

            OR if typing is needed for search:

            {
            'thought': 'Your thought process',
            'action': {
                'tool_name': 'type_tool',
                'tool_params': [
                {
                    'name': 'box_id',
                    'value': 22
                },
                {
                    'name': 'words',
                    'value': 'text to type'
                },
                {
                    'name': 'enter',
                    'value': 'true'
                }
                ]
            },
            'answer': 'Description of the action taken and the name of the element you interacted with, and the next step, as detailed as possible'
            }

            OR if you found the answer and believe the task is completed:

            {
            'thought': 'Your thought process',
            'action': {
                'tool_name': 'output_final_answer',
                'tool_params': [
                {
                    'name': 'answer',
                    'value': 'The answer is 123'
                }
                
                ]
            },
            'answer': 'used output_final_answer, task completed'
            }


            REMEMBER: Use proper JSON format with double quotes for all keys and string values. Do not use single quotes in your JSON output.
            """

            
            messages = [
        {
            "role": "user",
            "content": {
            "type": "image",
            "image": {
                "url": {
                "uri": encode_image(screenshot_path)
                }
            }
            }
        },
        {
            "role": "user",
            "content": prompt_instruction
        }
]
            response = agent.create_turn(
            messages=messages,
            session_id=session_id,
            stream=True,
            )
            action_str = ""
            for log in EventLogger().log(response):  
                action_str += log.content
                log.print()
            final_str = action_str.split('"answer": ')[-1].split('}')[0]
            self.action_history.append(final_str)


            # Optional: wait a short duration for UI to update after the tool call.
            time.sleep(2)
            self.step += 1


if __name__ == "__main__":
    agent = ComputerUseAgent(max_step=10)
    agent.run_agent()
