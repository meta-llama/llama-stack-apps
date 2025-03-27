import ast
import base64

import pyautogui
import requests
from gradio_client import Client, handle_file

# Initialize clients OUTSIDE the functions
# launch the OmniParser server locally and pass the URL here
OMNIPARSER_API_URL = ""

omni_client = Client(OMNIPARSER_API_URL)


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


def run_parser(image_path: str) -> str | None:
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
