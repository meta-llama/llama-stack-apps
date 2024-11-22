import pytest
from examples.interior_design_assistant.utils import enforce_response_format

def test_valid_output():
    output = """
    [
        {"description": "Modern leather sofa with sleek black finish"},
        {"description": "Oak wood coffee table with glass top"},
        {"description": "Elegant brass floor lamp with a white shade"}
    ]
    """
    result = enforce_response_format(output, 3)
    assert result == [
        "Modern leather sofa with sleek black finish",
        "Oak wood coffee table with glass top",
        "Elegant brass floor lamp with a white shade"
    ]

def test_invalid_json_output():
    output = "Invalid output"
    result = enforce_response_format(output, 3)
    assert len(result) == 3
    assert all("Alternative suggestion" in desc for desc in result)

def test_missing_description_field():
    output = """
    [
        {"not_description": "This is not valid"},
        {"not_description": "Neither is this"}
    ]
    """
    result = enforce_response_format(output, 3)
    assert len(result) == 3
    assert all("Alternative suggestion" in desc for desc in result)
