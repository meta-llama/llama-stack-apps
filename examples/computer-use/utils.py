import pyautogui
import requests

def run_parser(image_path: str, host = "localhost", port: int = 5000) -> dict[str, any):
    """
    Sends the screenshot image to the OmniParser server to obtain structured UI information.
    The OmniParser server is expected to return a dictionary with keys:
        - labeled_image: annotated/screenshot image information.
        - parsed_content_list: a list of dicts, each with "box_id", "label" and "coordinates" (x, y).
        - coordinate: (optional) overall coordinate info.
    If the server call fails, an empty output is returned.
    """
    server_url = f"http://{host}:{port}/omniparse"  # Change this to your OmniParser endpoint
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": img_file}
            response = requests.post(server_url, files=files)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print("Error contacting OmniParser server", e)
        raise e
        
def get_element_center(box_id: int, parsed_output: list[dict]) -> tuple[int, int] | None:
    """Finds element by box_id, returns center coordinates.

    Args:
        box_id: The ID of the element to find.
        parsed_output: The parsed output from OmniParser.

    Returns:
        A tuple containing the (x, y) coordinates of the center of the element,
        or None if the element is not found.
    """
    print(f"Searching for Box ID: {box_id}")  # Debug print 1
    # print(f"Available Box IDs: {[element['id'] for element in parsed_output]}")  # Debug print 2

    for element in parsed_output:
        if element['id'] == "icon " + str(box_id):
            x1, y1, x2, y2 = element['bbox']
            center_x = int(((x1 + x2) / 2) * pyautogui.size()[0])
            center_y = int(((y1 + y2) / 2) * pyautogui.size()[1])
            return center_x, center_y
    return None
