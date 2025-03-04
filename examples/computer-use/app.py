#!/usr/bin/env python
import os
import uuid
import re
import time
import requests
import pyautogui
import fire

# Import llama-stack client components
from llama_stack_client import LlamaStackClient
from llama_stack_client.lib.agents.client_tool import client_tool
from llama_stack_client.lib.agents.event_logger import EventLogger
from llama_stack_client.lib.agents.react.agent import ReActAgent

# Global variable to store the latest OmniParser output
latest_omni_output = {}

def get_som_labeled_img(image_path: str):
    """
    Sends the screenshot image to the OmniParser server to obtain structured UI information.
    The OmniParser server is expected to return a dictionary with keys:
        - labeled_image: annotated/screenshot image information.
        - parsed_content_list: a list of dicts, each with "box_id", "label" and "coordinates" (x, y).
        - coordinate: (optional) overall coordinate info.
    If the server call fails, a dummy output is returned.
    """
    server_url = "http://localhost:5000/omniparse"  # Change this to your OmniParser endpoint
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            response = requests.post(server_url, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print("Error contacting OmniParser server, using dummy output:", e)
        # Dummy output: simulate a parsed UI with two boxes.
        dummy = {
            "labeled_image": "dummy_labeled_image.png",
            "parsed_content_list": [
                {"box_id": "Box 1", "label": "clone", "coordinates": (100, 150)},
                {"box_id": "Box 2", "label": "llama-stack", "coordinates": (200, 250)}
            ],
            "coordinate": None,
        }
        return dummy

# ------------------ Tool Implementations ------------------

@client_tool
def click_tool(box_id: str):
    """
    Given a box_id, this tool looks up the coordinate from the latest OmniParser output and simulates a click.
    """
    global latest_omni_output
    if latest_omni_output and "parsed_content_list" in latest_omni_output:
        for element in latest_omni_output["parsed_content_list"]:
            if element["box_id"].lower() == box_id.lower():
                x, y = element["coordinates"]
                pyautogui.click(x, y)
                return f"Clicked {box_id} at {(x, y)}."
        return f"Box id {box_id} not found."
    else:
        return "No OmniParser output available."

@client_tool
def type_tool(box_id: str, words: str):
    """
    Given a box_id and a words string, this tool finds the coordinates from the OmniParser output,
    clicks to focus the corresponding UI element, and types the provided text.
    """
    global latest_omni_output
    if latest_omni_output and "parsed_content_list" in latest_omni_output:
        for element in latest_omni_output["parsed_content_list"]:
            if element["box_id"].lower() == box_id.lower():
                x, y = element["coordinates"]
                # Click to focus the UI element before typing.
                pyautogui.click(x, y)
                pyautogui.typewrite(words, interval=0.05)
                return f"Typed '{words}' in {box_id} at {(x, y)}."
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

def main(host: str = "localhost", port: int = 8000):
    """
    Sets up the Llama-Stack client and ReAct agent, gets a user query,
    and then iteratively uses the OmniParser server output and agent’s decision to drive UI actions.
    
    Workflow:
      1. Screenshot the current UI.
      2. Feed screenshot to OmniParser server via get_som_labeled_img().
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
        builtin_toolgroups=["builtin::websearch"],
        client_tools=[click_tool, type_tool, terminal_tool, homepage_tool],
        json_response_format=True,
    )
    
    session_id = agent.create_session(f"browser-agent-session-{uuid.uuid4().hex}")
    user_query = input("Enter your query (e.g., 'clone llama-stack repo'): ")
    print("Starting browser agent for query:", user_query)
    
    max_steps = 10
    step = 0
    while step < max_steps:
        print(f"\n---- Step {step + 1} ----")
        # Take a screenshot of the current screen.
        screenshot_path = f"temp_screenshot_{step}.png"
        screen_image = pyautogui.screenshot()
        screen_image.save(screenshot_path)
        print("Screenshot taken and saved as:", screenshot_path)
        
        # Call OmniParser with the screenshot to get structured UI info.
        global latest_omni_output
        latest_omni_output = get_som_labeled_img(screenshot_path)
        print("OmniParser output:", latest_omni_output)
        
        # Prepare the prompt for the agent that includes the user query and current UI state.
        omni_info = f"OmniParser Output: {latest_omni_output}"
        prompt_instruction = (
            f"User query: {user_query}\n"
            f"Current Screen Info: {omni_info}\n"
            "Determine next action. "
            "Respond with a tool-call command like 'click Box 1' or 'type Box 2: <text>'. "
            "If the goal is reached, output 'Goal reached, no action needed'."
        )
        
        # Send the turn to the agent; stream output if desired.
        response = agent.create_turn(
            messages=[{"role": "user", "content": prompt_instruction}],
            session_id=session_id,
            stream=True,
        )
        
        final_command = ""
        for log in EventLogger().log(response):
            final_command += log.message + "\n"
            print(log.message)
        
        # Check if the agent has signaled that the goal is reached.
        if "Goal reached" in final_command:
            print("Goal reached. Ending session.")
            break
        
        # Parse the agent's response to decide which tool to call.
        # Very simple parsing logic is used here.
        if "click" in final_command.lower():
            match = re.search(r"click\s+(Box\s*\d+)", final_command, re.IGNORECASE)
            if match:
                box_id = match.group(1).strip()
                tool_response = click_tool(box_id)
                print("Tool response:", tool_response)
            else:
                print("Could not parse click command.")
        elif "type" in final_command.lower():
            match = re.search(r"type\s+(Box\s*\d+)\s*:\s*(.+)", final_command, re.IGNORECASE)
            if match:
                box_id = match.group(1).strip()
                words = match.group(2).strip()
                tool_response = type_tool(box_id, words)
                print("Tool response:", tool_response)
            else:
                print("Could not parse type command.")
        elif "terminal" in final_command.lower():
            tool_response = terminal_tool()
            print("Tool response:", tool_response)
        elif "homepage" in final_command.lower():
            tool_response = homepage_tool()
            print("Tool response:", tool_response)
        else:
            print("No valid tool call found in agent response. Waiting briefly...")
        
        # Optional: wait a short duration for UI to update after the tool call.
        time.sleep(2)
        step += 1

if __name__ == "__main__":
    fire.Fire(main)
