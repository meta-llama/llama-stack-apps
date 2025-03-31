import logging
from llama_stack_client.lib.agents.client_tool import client_tool

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@client_tool
def calculator(x: float, y: float, operation: str) -> dict:
    """Simple calculator tool that performs basic math operations.

    :param x: First number to perform operation on
    :param y: Second number to perform operation on
    :param operation: Mathematical operation to perform ('add', 'subtract', 'multiply', 'divide')
    :returns: Dictionary containing success status and result or error message
    """
    logger.debug(f"Calculator called with: x={x}, y={y}, operation={operation}")
    try:
        if operation == "add":
            result = float(x) + float(y)
        elif operation == "subtract":
            result = float(x) - float(y)
        elif operation == "multiply":
            result = float(x) * float(y)
        elif operation == "divide":
            if float(y) == 0:
                return {"success": False, "error": "Cannot divide by zero"}
            result = float(x) / float(y)
        else:
            return {"success": False, "error": "Invalid operation"}

        logger.debug(f"Calculator result: {result}")
        return {"success": True, "result": result}
    except Exception as e:
        logger.error(f"Calculator error: {str(e)}")
        return {"success": False, "error": str(e)}
