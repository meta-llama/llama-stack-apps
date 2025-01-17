import os
from typing import Literal, Optional, Tuple, Union
from llama_stack_client import LlamaStackClient
from llama_models.llama3.api.chat_format import ChatFormat
from llama_models.llama3.api.tokenizer import Tokenizer
from llama_models.llama3.api.datatypes import StopReason
from llama_models.llama3.api.tool_utils import (
    is_valid_python_list,
    parse_python_list_for_function_calls,
)
import re
from llama_agent.utils.file_tree import list_files_in_repo
from llama_agent import REPO_DIR
from llama_agent.utils.ansi import red, yellow, magenta, blue
from subprocess import run

# Currently only supports 3.3-70B-Instruct at the moment since it depends on the 3.3/3.2 tool prompt format
MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
ITERATIONS = 15

SANDBOX_DIR = os.path.join(REPO_DIR, "sandbox")
# We give the agent a virtual working directory so it doesn't have to worry about long absolute paths
AGENT_WORKING_DIR = "/workspace/"

formatter = ChatFormat(Tokenizer.get_instance())


def run_agent(
    client: LlamaStackClient, repo: str, issue_title: str, issue_body: str
) -> Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
    """
    Returns:
        Tuple[Literal["changes_made", "no_changes_made"], str, Optional[str]]:
            ("changes_made", pr_title, pr_body): "changes_made", the PR title, and the PR body
            or ("no_changes_made", reasoning, None): "no_changes_made", the reason why no changes were made, and None
    """

    # System prompt
    message = "<|begin_of_text|>"
    message += header("system")
    message += """
    You are an expert software engineer.
    You will be given a problem statement in <problem_statement>

    Based on the <problem_statement>, you will need to make one or more function/tool calls to achieve the purpose.
    If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
    also point it out. You should only return the function call in tools call sections.

    If you decide to invoke any of the function(s), you MUST put it in the format of <tool>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]</tool>
    If you decide to invoke multiple functions, you MUST put commas between the function calls. E.g., <tool>[func_name1(params), func_name2(params), func_name3(params)]</tool>

    Here is a list of functions in JSON format that you can invoke.

    [
        {
            "name": "list_files",
            "description": "List all files in a directory.",
            "parameters": {
                "type": "dict",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a directory, e.g. `/workspace/django`. If referencing a file, will return the name of the file."
                    }
                },
            }
        },
        {
            "name": "edit_file",
            "description": "Edit a file. Specify the path to the file and the new_str to write to it. If old_str is specified, only the old_str will be replaced with new_str, otherwise the entire file will be replaced by new_str.",
            "parameters": {
                "type": "dict",
                "required": ["path", "new_str"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory, e.g. `/workspace/django/file.py` or `/workspace/django`."
                    },
                    "old_str": {
                        "type": "string",
                        "description": "The string in the file at `path` to replace. If not specified, the entire file will be replaced by new_str"
                    },
                    "new_str": {
                        "type": "string",
                        "description": "The new string to write to the file. If the old_str is specified, only the old_str will be replaced with new_str, otherwise the entire file will be replaced by new_str."
                    }
                }
            }
        },
        {
            "name": "view_file",
            "description": "View a file",
            "parameters": {
                "type": "dict",
                "required": ["path"],
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The absolute path to the file to view, e.g. `/workspace/django/file.py` or `/workspace/django`."
                    }
                }
            }
        },
        {
            "name": "finish",
            "description": "If you have solved the problem, you can call this function to finish the task.",
            "parameters": {}
        }
    ]

    Please explain your reasoning before you make any edits in a <thinking> tag.

    <|eot_id|>
    """.strip()

    # User prompt
    message += header("user")
    files_in_repo = "\n".join(
        list_files_in_repo(os.path.join(SANDBOX_DIR, repo), depth=2)
    )
    message += f"""
    <working_directory>
    {os.path.join(AGENT_WORKING_DIR, repo)}
    </working_directory>

    <file_tree>
    {files_in_repo}
    </file_tree>

    <problem_statement>
    Issue title: {issue_title}
    Issue body: {issue_body}
    </problem_statement>

    You are in the working directory as specified in <working_directory>. Please specify paths in absolute paths only.
    I have included the top level files and directories in the repository in <file_tree>.
    Please start by listing out and viewing files in the repository to understand the problem.<|eot_id|>
    """.strip()

    finished = False
    for i in range(ITERATIONS):
        print("\n")
        print(f"Iteration {i+1} of {ITERATIONS}")
        print("-" * 80)

        if finished:
            break

        message += header("assistant")
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )

        # Display thinking alongside with tool calls
        thinking_match = re.search(
            r"<thinking>(.*?)</thinking>", response.content, re.DOTALL
        )
        if thinking_match:
            print(f"Thinking: {magenta(thinking_match.group(1).strip())}")
        else:
            # Check for any text outside of tool tags
            non_tool_content = re.sub(
                r"<tool>.*?</tool>", "", response.content, flags=re.DOTALL
            ).strip()
            if non_tool_content:
                print(f"Thinking: {magenta(non_tool_content)}")

        message += response.content
        message += f"<|eot_id|>"

        # Evaluate tool calls
        tool_calls = parse_tool_calls(response.content)
        for tool_call in tool_calls:

            if tool_call[0] == "error":
                _, error_message = tool_call
                msg = f"ERROR - Could not parse tool call: {error_message}"
                print(red(msg))
                message += chat_message("tool", msg)
                continue

            tool_name, tool_params = tool_call
            msg = (
                f"Executing tool call: "
                + blue(f"[{tool_name}{display_tool_params(tool_params)}]")
            )
            message += header("tool")
            message += msg + "\n"
            print(msg)

            try:
                result, result_msg = execute_tool_call(tool_name, tool_params)
            except Exception as e:
                result, result_msg = ("error", f"ERROR - Calling tool: {tool_name} {e}")

            message += f"Result: {result_msg}\n"

            if result == "success":
                # Truncate the result message to 200 characters since it can be long
                print("Result: " + result_msg[:200] + "...")
            else:
                print("Result: " + result_msg)

            message += f"<|eot_id|>"

            if result == "success" and tool_name == "finish":
                finished = True

    if finished:
        print(blue("Agent marked as finished"))
    else:
        print(yellow("Max iterations reached"))

    # Create a PR title
    message += chat_message(
        "user",
        "Please create a PR title that summarizes the changes you've made. Do not include any leading or trailing punctuation.",
    )
    message += header("assistant")
    response = client.inference.completion(
        model_id=MODEL_ID,
        content=message,
    )
    pr_title = response.content

    # Check if there are any changes
    # If there are no changes, ask the agent to explain why
    diff_cmd = run(f"cd {os.path.join(SANDBOX_DIR, repo)} && git diff", shell=True, capture_output=True)
    if not diff_cmd.stdout:
        print(f"No changes were made - agent explaining why...")
        message += chat_message(
            "user",
            (
                "No changes were made."
                "Could you explain your reasoning for not making any changes?"
                "Please write it in GitHub Flavored Markdown."
                "Also provide some next steps to fix the issue."
            ),
        )
        message += header("assistant")
        response = client.inference.completion(
            model_id=MODEL_ID,
            content=message,
        )
        reasoning = response.content
        return ("no_changes_made", reasoning, None)

    # Create a PR body
    message += chat_message(
        "user",
        (
            "Summarizing all of the changes and thinking you've done,"
            "please write a PR body that explains the changes you've made."
            "Please write it in GitHub Flavored Markdown."
        ),
    )
    message += header("assistant")
    # Llama sometimes includes an unnecessary "## PR body" title so we add it here to make sure it's not included
    message += "## PR Body\n\n"
    response = client.inference.completion(
        model_id=MODEL_ID,
        content=message,
    )
    pr_body = response.content

    return "changes_made", pr_title, pr_body


def execute_tool_call(
    tool_name: str, tool_params: dict[str, str]
) -> Union[Tuple[Literal["success"], str], Tuple[Literal["error"], str]]:
    """
    Execute a tool call and return a message indicating the result of the tool call.

    Args:
        tool_name (str): The name of the tool to execute.
        tool_params (dict[str, str]): The parameters to pass to the tool.

    Returns:
        Union[Tuple[Literal["success"], str], Tuple[Literal["error"], str]]:
            ("success", result): The result of the tool call.
            ("error", error_message): The error message if the tool call failed.
    """
    if tool_name == "list_files":
        if (error := validate_param_exists("path", tool_params)
            or validate_not_symlink(tool_params["path"])
            or validate_path_in_sandbox(tool_params["path"])
            or validate_directory_exists(tool_params["path"])):
            return ("error", error)

        path = translate_path(tool_params["path"])
        files = list_files_in_repo(path, depth=1)
        return ("success", "\n".join(files))

    elif tool_name == "edit_file":
        if (
            error := validate_param_exists("path", tool_params)
            or validate_path_in_sandbox(tool_params["path"])
            or validate_param_exists("new_str", tool_params)
            or validate_not_symlink(tool_params["path"])
            or validate_file_exists(tool_params["path"])
            or validate_not_a_directory(tool_params["path"])
        ):
            return ("error", error)

        path = translate_path(tool_params["path"])
        if "old_str" in tool_params:
            with open(f"{path}", "r") as f:
                file_content = f.read()
            with open(f"{path}", "w") as f:
                old_str = tool_params["old_str"]
                new_str = tool_params["new_str"]
                new_content = file_content.replace(old_str, new_str)
                f.write(new_content)
        else:
            with open(f"{path}", "w") as f:
                f.write(tool_params["new_str"])
        return ("success", "File successfully updated")


    elif tool_name == "view_file":
        if (error := validate_param_exists("path", tool_params)
            or validate_not_symlink(tool_params["path"])
            or validate_path_in_sandbox(tool_params["path"])
            or validate_file_exists(tool_params["path"])
            or validate_not_a_directory(tool_params["path"])):
            return ("error", error)

        path = translate_path(tool_params["path"])
        with open(f"{path}", "r") as f:
            file_content = f.read()
        return ("success", file_content)

    elif tool_name == "finish":
        return ("success", "Task marked as finished")

    else:
        return ("error", f"ERROR - Unknown tool: {tool_name}")


def translate_path(path: str) -> str:
    if path.startswith(AGENT_WORKING_DIR):
        return os.path.join(SANDBOX_DIR, path[len(AGENT_WORKING_DIR) :])
    else:
        return os.path.join(SANDBOX_DIR, path)


def parse_tool_calls(
    content,
) -> list[Union[tuple[str, dict[str, str]], tuple[Literal["error"], str]]]:
    """
    Parse tool calls from the content.

    Args:
        content (str): The content to parse tool calls from.

    Returns:
        list[Union[tuple[str, dict[str, str]], tuple[Literal["error"], str]]: Either:
            tuple[str, dict[str, str]]:
                - name (str): The name of the tool
                - params (dict): The parameters of the tool
            or tuple[Literal["error"], str] if the tool call could not be parsed:
                - "error"
                - error_message (str): The error message
    """
    tool_calls = []
    for match in re.finditer(r"<tool>(.*?)</tool>", content, re.DOTALL):
        tool_content = match.group(1)
        if not is_valid_python_list(tool_content):
            tool_content = tool_content.strip()

            # Add square brackets if missing
            if not tool_content.startswith("["):
                tool_content = f"[{tool_content}"
            if not tool_content.endswith("]"):
                tool_content = f"{tool_content}]"

        try:
            result = parse_python_list_for_function_calls(tool_content)
            if is_valid_python_list(tool_content):
                # Add the original tool content to each result tuple
                result = [(name, params) for name, params in result]
                tool_calls.extend(result)
            else:
                tool_calls.append(
                    (
                        "error",
                        "Tool call invalid syntax: " + match.group(0),
                    )
                )
        except Exception as e:
            tool_calls.append(
                (
                    "error",
                    "Tool call invalid syntax: Could not parse tool call: "
                    + match.group(0)
                    + " "
                    + str(e),
                )
            )

    return tool_calls


def display_tool_params(tool_params: dict[str, str]):
    return (
        "("
        + ", ".join(
            [
                param_name + '="' + str(param_value) + '"'
                for param_name, param_value in tool_params.items()
            ]
        )
        + ")"
    )


def validate_param_exists(
    param_name: str, tool_params: dict[str, str]
) -> Optional[str]:
    if param_name not in tool_params:
        return f"ERROR - {param_name} not found in tool params: {display_tool_params(tool_params)}"
    return None

def validate_path_in_sandbox(path: str) -> Optional[str]:
    """
    Validate that a path stays within the sandbox directory.
    
    Args:
        path (str): The path to validate

    Returns:
        Optional[str]: Error message if path is invalid, None if valid
    """
    # Resolve the absolute path after translation to catch any ../ tricks
    path = translate_path(path)
    resolved_path = os.path.abspath(path)
    sandbox_path = os.path.abspath(SANDBOX_DIR)

    if not resolved_path.startswith(sandbox_path):
        # From the agent's perspective, any paths not in the sandbox don't exist
        return f"ERROR - File {path} does not exist"
    return None

def validate_not_symlink(path: str) -> Optional[str]:
    if os.path.islink(translate_path(path)):
        return f"ERROR - File {path} is a symlink. Simlinks not allowed"
    return None

def validate_file_exists(path: str) -> Optional[str]:
    if not os.path.exists(translate_path(path)):
        return f"ERROR - File {path} does not exist. Please ensure the path is an absolute path and that the file exists."
    return None

def validate_not_a_directory(path: str) -> Optional[str]:
    if os.path.isdir(translate_path(path)):
        return f"ERROR - File {path} is a directory. Please ensure the path references a file, not a directory."
    return None

def validate_directory_exists(path: str) -> Optional[str]:
    if not os.path.exists(translate_path(path)):
        return f"ERROR - Directory {path} does not exist. Please ensure the path is an absolute path and that the directory exists."
    return None

def chat_message(role: Literal["user", "assistant", "system", "tool"], content: str):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"


def header(role: Literal["user", "assistant", "system", "tool"]):
    return f"<|start_header_id|>{role}<|end_header_id|>\n\n"
