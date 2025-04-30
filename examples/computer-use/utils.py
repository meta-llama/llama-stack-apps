# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import base64
import ast
import pyautogui
import requests
from PIL import Image, ImageDraw, ImageFont
from gradio_client import Client, handle_file

import uuid
from typing import List, Optional, Union, Dict, Any
import re
from pydantic import BaseModel, ValidationError

from llama_stack_client.types.shared.completion_message import CompletionMessage
from llama_stack_client.types.shared.tool_call import ToolCall
from llama_stack_client.lib.agents.tool_parser import ToolParser

import json
# Initialize clients OUTSIDE the functions
import os
from PIL import ImageFont

def load_first_font_from_library(size=12,font_dir="/Library/Fonts/"):
    """
    Finds the first loadable TrueType (.ttf), OpenType (.otf), or 
    TrueType Collection (.ttc) font within /Library/Fonts/ using os.walk().

    Args:
        size (int): The desired font size. Defaults to 12.

    Returns:
        PIL.ImageFont.FreeTypeFont or None: The loaded font object, 
                                             or None if no suitable fonts are found, 
                                             the directory doesn't exist, or an error 
                                             occurs during loading.
    """
    allowed_extensions = ('.ttf', '.otf', '.ttc')

    # Check if the target directory exists [13]
    if not os.path.isdir(font_dir):
        print(f"Error: Font directory '{font_dir}' not found or is not a directory.")
        return None

    try:
        # Walk the directory tree [3][5]
        for root, dirs, files in os.walk(font_dir, topdown=True): 
            # Sort files for slightly more predictable behavior, though os.walk order isn't guaranteed [1]
            files.sort()  
            for filename in files:
                # Check for allowed font file extensions
                if filename.lower().endswith(allowed_extensions):
                    # Construct the full path to the font file [5][15]
                    font_path = os.path.join(root, filename)
                    try:
                        # Attempt to load the font using PIL [6][8][10][17]
                        font = ImageFont.truetype(font_path, size=size)
                        # print(f"Successfully loaded font: {font_path}") # Optional: confirmation
                        return font # Return the first font successfully loaded
                    except (IOError, OSError, ValueError) as e:
                        # Catch errors during font loading (e.g., corrupted file, permission issues) [6][8]
                        # ValueError can occur for some font issues too.
                        # print(f"Warning: Could not load font '{font_path}'. Reason: {e}. Trying next...") 
                        continue # Move to the next file

    except OSError as e:
        # Catch errors during directory traversal (e.g., permission denied) [7]
        print(f"Error walking directory '{font_dir}': {e}")
        return None

    # If the loop completes without returning a font
    print(f"No loadable font files with extensions {allowed_extensions} found in '{font_dir}' or its subdirectories.")
    return None

def get_ruled_screenshot(screenshot_path):

    image = pyautogui.screenshot()
    # Get the image dimensions
    width, height = image.size
    print(f"Image dimensions: {width}x{height}")
    print(pyautogui.size().width, pyautogui.size().height)
    if width != pyautogui.size().width or height != pyautogui.size().height:
        print("Resizing image to match screen dimensions")
        image = image.resize((pyautogui.size().width, pyautogui.size().height))
        width, height = image.size
        print(f"Image dimensions after resizing: {width}x{height}")
    # Create a new image for the semi-transparent layer
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # Transparent layer
    draw = ImageDraw.Draw(overlay)

    # Set the line color (gray) and line opacity (adjusting the alpha value)
    line_color = (250, 0, 0, 128)  # The last value (128) controls opacity, 0 = fully transparent, 255 = fully opaque

    # Load a font for the labels (you can specify any TTF font you have)
    try:
        font = load_first_font_from_library(size=16)
    except IOError:
        font = ImageFont.load_default()

    # Draw vertical and horizontal lines every 100 pixels and add labels
    for x in range(0, width, 100):
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
        # Add labels at the top for vertical lines
        if x % 100 == 0:
            draw.text((x + 5, 5), str(x), font=font, fill=(250, 0, 0, 128))
            draw.text((x, height - 25), str(x), font=font, fill=(250, 0, 0, 128))

    for y in range(0, height, 100):
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
        # Add labels on the left for horizontal lines
        if y % 100 == 0:
            draw.text((5, y + 5), str(y), font=font, fill=(250, 0, 0, 128))
            text_width, text_height = 35, 15
            draw.text((width - text_width - 5, y + 5), str(y), font=font, fill=(250, 0, 0, 128))

    # Convert screenshot to RGBA for proper merging
    image = image.convert("RGBA")

    # Merge the overlay (with lines and labels) back onto the original image
    combined = Image.alpha_composite(image.convert("RGBA"), overlay)
    combined.save(screenshot_path)
class Param(BaseModel):
    name: str
    value: Union[str, int, float, bool]


class Action(BaseModel):
    tool_name: str
    tool_params: List[Param]


class ReActOutput(BaseModel):
    thought: str
    action: Optional[Action]
    answer: Optional[str]
def parse_react_output(response_text: str) -> ReActOutput:
    """
    A robust parser that converts various text formats into a ReActOutput object.

    Args:
        response_text: String containing thought, action, and answer in various formats

    Returns:
        ReActOutput: A properly formatted ReActOutput object
    """
    # First attempt: try to parse as valid JSON
    try:
        data = json.loads(response_text)
        return process_parsed_data(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Second attempt: try to clean the text and parse as JSON
    cleaned_text = clean_text(response_text)
    try:
        data = json.loads(cleaned_text)
        return process_parsed_data(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Third attempt: try to fix structural issues in the JSON
    fixed_text = fix_json_structure(cleaned_text)
    try:
        data = json.loads(fixed_text)
        return process_parsed_data(data)
    except (json.JSONDecodeError, ValueError):
        pass

    # Final attempt: fall back to regex-based extraction
    return extract_with_regex(response_text)

def clean_text(text: str) -> str:
    """Clean up the text to make it proper JSON."""
    # Remove leading/trailing whitespace and newlines
    text = text.strip()

    # If text is enclosed in triple quotes, remove them
    if text.startswith('"""') and text.endswith('"""'):
        text = text[3:-3].strip()

    # Replace Python None/null with JSON null
    text = re.sub(r'\bNone\b', 'null', text)

    # Replace Python style single quotes with double quotes
    # First, temporarily replace already escaped quotes to preserve them
    text = text.replace("\\'", "___ESCAPED_SINGLE_QUOTE___")
    text = text.replace('\\"', "___ESCAPED_DOUBLE_QUOTE___")

    # Replace single quotes with double quotes
    text = text.replace("'", '"')

    # Restore escaped quotes
    text = text.replace("___ESCAPED_SINGLE_QUOTE___", "\\'")
    text = text.replace("___ESCAPED_DOUBLE_QUOTE___", '\\"')

    # Replace Python booleans with JSON booleans
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)

    # Ensure the text is wrapped in curly braces
    if not text.startswith('{'):
        text = '{' + text
    if not text.endswith('}'):
        text = text + '}'

    return text

def fix_json_structure(text: str) -> str:
    """Fix common structural issues in JSON-like strings."""
    # Fix missing commas between properties
    text = re.sub(r'"}(\s*)"', '"},\\1"', text)

    # Fix missing colons after property names
    text = re.sub(r'"(\w+)"\s+([{\["0-9tfn])', r'"\1": \2', text)

    # Fix tool_params structure when it's in dictionary format
    tool_params_dict = re.search(r'"tool_params":\s*\{([^}]+)\}', text)
    if tool_params_dict:
        dict_content = tool_params_dict.group(1)
        params_list = []

        for kv_match in re.finditer(r'"([^"]+)":\s*([^,}]+)', dict_content):
            name = kv_match.group(1)
            value = kv_match.group(2).strip()
            params_list.append(f'{{"name": "{name}", "value": {value}}}')

        if params_list:
            list_replacement = f'"tool_params": [{", ".join(params_list)}]'
            text = re.sub(r'"tool_params":\s*\{[^}]+\}', list_replacement, text)

    # Fix missing braces around the action (example 4 issue)
    action_match = re.search(r'"action":\s*"tool_name":', text)
    if action_match:
        pos = action_match.start() + len('"action":')
        text = text[:pos] + ' {' + text[pos:]

        # Find where to insert the closing brace
        answer_pos = text.find('"answer":', pos)
        if answer_pos != -1:
            text = text[:answer_pos] + '},' + text[answer_pos:]
        else:
            # If no answer field, insert before the closing brace of the whole object
            text = text[:-1] + '}' + text[-1:]

    # Balance braces and brackets
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # Add missing closing braces/brackets
    if open_braces > close_braces:
        text = text[:-1] + ('}' * (open_braces - close_braces)) + text[-1:]
    if open_brackets > close_brackets:
        text = text[:-1] + (']' * (open_brackets - close_brackets)) + text[-1:]

    return text

def process_parsed_data(data: Dict[str, Any]) -> ReActOutput:
    """Process parsed JSON data into a ReActOutput object."""
    thought = data.get('thought', '')

    # Handle answer field
    answer_raw = data.get('answer')
    answer = None if answer_raw in (None, 'None', 'null') else str(answer_raw)

    # Handle action field
    action_data = data.get('action')
    action = None

    if action_data:
        tool_name = action_data.get('tool_name', '')
        tool_params_data = action_data.get('tool_params', [])
        tool_params = []

        # Handle tool_params as a dictionary (example 1)
        if isinstance(tool_params_data, dict):
            for name, value in tool_params_data.items():
                tool_params.append(Param(name=name, value=value))

        # Handle tool_params as a list (examples 2-4)
        elif isinstance(tool_params_data, list):
            for param in tool_params_data:
                if isinstance(param, dict) and 'name' in param and 'value' in param:
                    tool_params.append(Param(name=param['name'], value=param['value']))

        if tool_name:
            action = Action(tool_name=tool_name, tool_params=tool_params)

    return ReActOutput(thought=thought, action=action, answer=answer)

def extract_with_regex(text: str) -> ReActOutput:
    """Extract fields using regex when all other parsing methods fail."""
    # Extract thought
    thought_match = re.search(r'["\']thought["\']:\s*["\']([^"\']+)["\']', text, re.DOTALL)
    thought = thought_match.group(1) if thought_match else ""

    # Extract answer
    answer_match = re.search(r'["\']answer["\']:\s*(?:["\']([^"\']+)["\']|null|None)', text)
    answer = None
    if answer_match:
        answer_value = answer_match.group(1)
        if answer_value and answer_value not in ('None', 'null'):
            answer = answer_value

    # Extract action
    action = None
    tool_name_match = re.search(r'["\']tool_name["\']:\s*["\']([^"\']+)["\']', text)

    if tool_name_match:
        tool_name = tool_name_match.group(1)
        params = []

        # Try to extract tool_params in list format
        param_matches = re.finditer(r'["\']name["\']:\s*["\']([^"\']+)["\'].*?["\']value["\']:\s*([^,\}\]]+)', text)
        for match in param_matches:
            name = match.group(1)
            value_str = match.group(2).strip()

            # Strip quotes if present
            if (value_str.startswith('"') and value_str.endswith('"')) or \
               (value_str.startswith("'") and value_str.endswith("'")):
                value_str = value_str[1:-1]

            # Convert value to appropriate type
            if value_str.lower() in ('true', 'false'):
                value = value_str.lower() == 'true'
            elif value_str.isdigit():
                value = int(value_str)
            elif value_str.replace('.', '', 1).isdigit():
                value = float(value_str)
            else:
                value = value_str

            params.append(Param(name=name, value=value))

        # If no params found in list format, try dictionary format
        if not params:
            dict_match = re.search(r'["\']tool_params["\']:\s*\{([^\}]+)\}', text)
            if dict_match:
                dict_content = dict_match.group(1)
                kv_matches = re.finditer(r'["\']([^"\']+)["\']:\s*([^,\}]+)', dict_content)

                for match in kv_matches:
                    name = match.group(1)
                    value_str = match.group(2).strip()

                    # Strip quotes if present
                    if (value_str.startswith('"') and value_str.endswith('"')) or \
                       (value_str.startswith("'") and value_str.endswith("'")):
                        value_str = value_str[1:-1]

                    # Convert value to appropriate type
                    if value_str.lower() in ('true', 'false'):
                        value = value_str.lower() == 'true'
                    elif value_str.isdigit():
                        value = int(value_str)
                    elif value_str.replace('.', '', 1).isdigit():
                        value = float(value_str)
                    else:
                        value = value_str

                    params.append(Param(name=name, value=value))

        if tool_name and params:
            action = Action(tool_name=tool_name, tool_params=params)

    return ReActOutput(thought=thought, action=action, answer=answer)

class ComputerToolParser(ToolParser):
    """A tool parser for the Computer Use domain."""

    def get_tool_calls(self, output_message: CompletionMessage) -> List[ToolCall]:
        tool_calls = []
        response_text = str(output_message.content)
        if "{" not in response_text:
            print(f"Error parsing action: not a valid JSON as {response_text}")
            return tool_calls
        response_text.split("{")[0]
        # Remove the first part of the response that is not JSON
        response_list = response_text.split("{")[1:]
        clean_text = '{' + "{".join(response_list)
        if clean_text[-1] != "}":
            clean_text += "}"
        #print(f"Clean text: {clean_text}")
        try:
            react_output = ReActOutput.model_validate_json(clean_text)
        except ValidationError as e:
            print('failed to parse, trying to fix')
            react_output = parse_react_output(clean_text)
            # print(f"Error parsing action: {e}")
            # print(f"Response text: {response_text}")
            print(react_output)
        # if react_output.answer:
        #     return tool_calls
        if react_output.action:
            tool_name = react_output.action.tool_name
            tool_params = react_output.action.tool_params
            params = {param.name: param.value for param in tool_params}
            if tool_name and tool_params:
                call_id = str(uuid.uuid4())
                tool_calls = [ToolCall(call_id=call_id, tool_name=tool_name, arguments=params)]
        return tool_calls

# def run_parser(image_path: str, host = "localhost", port: int = 5000) -> dict[str, any):
#     """
#     Sends the screenshot image to the OmniParser server to obtain structured UI information.
#     The OmniParser server is expected to return a dictionary with keys:
#         - labeled_image: annotated/screenshot image information.
#         - parsed_content_list: a list of dicts, each with "box_id", "label" and "coordinates" (x, y).
#         - coordinate: (optional) overall coordinate info.
#     If the server call fails, an empty output is returned.
#     """
#     server_url = f"http://{host}:{port}/omniparse"  # Change this to your OmniParser endpoint
#     try:
#         with open(image_path, "rb") as img_file:
#             files = {"file": img_file}
#             response = requests.post(server_url, files=files)
#             response.raise_for_status()
#             return response.json()
#     except Exception as e:
#         print("Error contacting OmniParser server", e)
#         raise e
def convert_to_base64(image_path: str) -> str:
    """Converts an image to base64.

    Args:
        image_path: The path to the image.

    Returns:
        The base64 encoded image as a string.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def run_parser(omni_client,image_path: str) -> str | None:
    """Calls OmniParser API, returns raw result.

    Args:
        image_path: The path to the image to process.

    Returns:
        The raw result string from OmniParser, or None if an error occurred.
    """
    try:
        result = omni_client.predict(
            image_input=handle_file(image_path),
            box_threshold=0.05,
            iou_threshold=0.1,
            use_paddleocr=True,
            imgsz=640,
            api_name="/process",
        )
        return result
    except Exception as e:
        print(f"Error calling OmniParser: {e}")
        return None


def parse_omni_parser_output(omni_parser_output_string: str) -> list[dict]:
    """
    Parses complex OmniParser output with error resilience.

    Args:
        omni_parser_output_string: The raw string output from OmniParser.

    Returns:
        A list of dictionaries, each representing a parsed element.  Returns
        an empty list on critical parsing errors.
    """
    parsed_content_list = []
    try:
        lines = omni_parser_output_string.strip().split("\n")
        for line_num, line in enumerate(lines, 1):
            # Split line into prefix and JSON part
            if ":" not in line:
                print(
                    f"Skipping line {line_num}: No colon found. Line: '{line}'", "bot"
                )
                continue
            prefix, json_str = line.split(":", 1)
            prefix, json_str = prefix.strip(), json_str.strip()

            # Parse JSON using ast.literal_eval (handles single quotes)
            try:
                content_dict = ast.literal_eval(json_str)
                if not isinstance(content_dict, dict):
                    print(
                        f"Skipping line {line_num}: Not a dictionary. Line: '{line}'",
                        "bot",
                    )
                    continue
            except (SyntaxError, ValueError) as e:
                print(
                    f"Skipping line {line_num}: Invalid syntax. Error: {e}. Line: '{line}'",
                    "bot",
                )
                continue

            # Validate required fields
            required_keys = {"type", "bbox", "interactivity", "content"}
            missing_keys = required_keys - content_dict.keys()
            if missing_keys:
                print(
                    f"Skipping line {line_num}: Missing keys {missing_keys}. Line: '{line}'",
                    "bot",
                )
                continue

            # Add metadata (optional: include prefix as 'id')
            content_dict["id"] = prefix  # e.g., "icon 0"
            parsed_content_list.append(content_dict)

    except Exception as e:
        print(f"Critical error parsing OmniParser output: {e}", "bot")
        return []
    return parsed_content_list


def get_element_center(
    box_id: int, parsed_output: list[dict]
) -> tuple[int, int] | None:
    """Finds element by box_id, returns center coordinates.

    Args:
        box_id: The ID of the element to find.
        parsed_output: The parsed output from OmniParser.

    Returns:
        A tuple containing the (x, y) coordinates of the center of the element,
        or None if the element is not found.
    """
    box_id = int(box_id)
    print(f"Searching for Box ID: {box_id}")  # Debug print 1
    # print(f"Available Box IDs: {[element['id'] for element in parsed_output]}")  # Debug print 2

    for element in parsed_output:
        if element["id"] == "icon " + str(box_id):
            x1, y1, x2, y2 = element["bbox"]
            center_x = int(((x1 + x2) / 2) * pyautogui.size()[0])
            center_y = int(((y1 + y2) / 2) * pyautogui.size()[1])
            return center_x, center_y
    return None
