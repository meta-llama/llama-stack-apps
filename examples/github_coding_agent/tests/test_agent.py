import pytest
from subprocess import run
from llama_agent.agent import (
    display_tool_params,
    parse_tool_calls,
    translate_path,
    SANDBOX_DIR,
    execute_tool_call,
    REPO_DIR,
)
from llama_agent.utils.file_tree import list_files_in_repo
import tempfile
import os
import shutil


class TestDisplayToolParams:
    def test_no_params(self):
        assert display_tool_params({}) == "()"

    def test_one_param(self):
        assert display_tool_params({"a": "b"}) == '(a="b")'

    def test_three_params(self):
        assert (
            display_tool_params({"a": "b", "c": "d", "e": "f"})
            == '(a="b", c="d", e="f")'
        )


class TestParseToolCallFromContent:
    def test_basic_tool_call(self):
        content = '<tool>[func1(a="1", b="2")]</tool>'
        assert parse_tool_calls(content) == [("func1", {"a": "1", "b": "2"})]

    def test_empty_arg(self):
        content = '<tool>[func1(a="1", b=)]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        error, error_message = res[0]
        assert error == "error"
        assert "Tool call invalid syntax" in error_message

    def test_handles_missing_left_matching_bracket(self):
        content = "<tool>func1()]</tool>"

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {}

    def test_handles_missing_right_matching_bracket(self):
        content = '<tool>[func1(a="1", b="2")]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {"a": "1", "b": "2"}

    def test_handles_missing_left_matching_bracket_and_right_matching_bracket(self):
        content = '<tool>func1(a="1", b="2")</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 1
        tool_name, tool_params = res[0]
        assert tool_name == "func1"
        assert tool_params == {"a": "1", "b": "2"}

    def test_handles_multiple_tool_calls(self):
        content = '<tool>[func1(a="1", b="2"), func2(c="3", d="4")]</tool>'

        res = parse_tool_calls(content)

        assert len(res) == 2
        assert res[0] == ("func1", {"a": "1", "b": "2"})
        assert res[1] == ("func2", {"c": "3", "d": "4"})

    def test_handles_multiple_tool_tags_and_text(self):
        content = """
        I should use func1 to do something.
        <tool>[func1(a="1", b="2")]</tool>
        I should use func2 to do something else.
        <tool>[func2(c="3", d="4")]</tool>
        """

        res = parse_tool_calls(content)

        assert len(res) == 2
        assert res[0] == ("func1", {"a": "1", "b": "2"})
        assert res[1] == ("func2", {"c": "3", "d": "4"})


class TestFileTree:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method"""
        # Create a temporary directory for tests
        # Create a new temp directory for each test
        self.test_dir = tempfile.mkdtemp()

        yield

        # Clean up the temp directory after each test
        shutil.rmtree(self.test_dir)

    def test_file_not_found(self):
        with pytest.raises(
            FileNotFoundError, match="File /workspace/does_not_exist does not exist"
        ):
            list_files_in_repo("/workspace/does_not_exist")

    def test_handles_if_file_is_not_in_git(self):
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()

        with pytest.raises(AssertionError, match="not a git repository"):
            list_files_in_repo(self.test_dir)

    def test_default_depth_1(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir1", "file2.txt"), "w").close()
        add_to_git(self.test_dir)

        res = list_files_in_repo(self.test_dir)

        assert res == ["dir1/", "file1.txt"]

    def test_depth_2(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "file1.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir1", "file2.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir2", "file3.txt"), "w").close()
        add_to_git(self.test_dir)

        res = list_files_in_repo(self.test_dir, depth=2)

        assert res == [
            "dir1/",
            "dir1/file2.txt",
            "dir2/",
            "dir2/file3.txt",
            "file1.txt",
        ]


class TestTranslatePath:

    def test_workspace_path(self):
        assert translate_path("/workspace/repo") == os.path.join(SANDBOX_DIR, "repo")

    def test_relative_path(self):
        assert translate_path("repo") == os.path.join(SANDBOX_DIR, "repo")


class TestExecuteToolCall:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test environment before each test method"""
        self.test_dir = os.path.join(SANDBOX_DIR, "test_repo")
        os.makedirs(self.test_dir)
        with open(os.path.join(self.test_dir, "file.txt"), "w") as f:
            f.write("old content\n\nHello World")

        yield

        shutil.rmtree(self.test_dir)

    def test_list_files_no_path(self):
        res = execute_tool_call("list_files", {})

        assert res == ("error", "ERROR - path not found in tool params: ()")

    def test_list_files_success(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        os.makedirs(os.path.join(self.test_dir, "dir2"))
        open(os.path.join(self.test_dir, "dir1", "file.txt"), "w").close()
        open(os.path.join(self.test_dir, "dir2", "file.txt"), "w").close()
        add_to_git(self.test_dir)

        res = execute_tool_call("list_files", {"path": "/workspace/test_repo"})

        assert res == ("success", "dir1/\ndir2/\nfile.txt")

    def test_list_files_throws_if_path_not_in_sandbox(self):
        path = "/workspace/../llama_agent"
        res = execute_tool_call("list_files", {"path": path})

        assert res == (
            "error",
            # From the agent's perspective, any paths not in the sandbox don't exist
            f"ERROR - File {SANDBOX_DIR}/../llama_agent does not exist",
        )

    def test_list_files_path_not_exists(self):
        res = execute_tool_call(
            "list_files", {"path": "/workspace/test_repo/does_not_exist"}
        )

        assert res == (
            "error",
            f"ERROR - Directory /workspace/test_repo/does_not_exist does not exist. Please ensure the path is an absolute path and that the directory exists.",
        )

    def test_list_files_relative_path(self):
        os.makedirs(os.path.join(self.test_dir, "dir1"))
        open(os.path.join(self.test_dir, "dir1", "file.txt"), "w").close()
        add_to_git(self.test_dir)

        res = execute_tool_call("list_files", {"path": "test_repo/dir1"})

        assert res == ("success", "file.txt")

    def test_list_files_symlink(self):
        os.symlink(REPO_DIR, os.path.join(self.test_dir, "bad_dir"))

        res = execute_tool_call("list_files", {"path": "/workspace/test_repo/bad_dir"})

        assert res == (
            "error",
            "ERROR - File /workspace/test_repo/bad_dir is a symlink. Simlinks not allowed",
        )

    def test_edit_file_success(self):
        res = execute_tool_call(
            "edit_file",
            {"path": "/workspace/test_repo/file.txt", "new_str": "new content"},
        )

        assert res == ("success", "File successfully updated")
        self.assert_file_content("file.txt", "new content")

    def test_edit_file_error(self):
        res = execute_tool_call(
            "edit_file", {"path": "repo/file.txt", "new_str": "new content"}
        )

        error, error_message = res
        assert error == "error"
        assert (
            "ERROR - File repo/file.txt does not exist. Please ensure the path is an absolute path and that the file exists."
            == error_message
        )

    def test_edit_file_no_path(self):
        res = execute_tool_call("edit_file", {"new_str": "new content"})

        assert res == (
            "error",
            'ERROR - path not found in tool params: (new_str="new content")',
        )

    def test_edit_file_no_new_str(self):
        res = execute_tool_call("edit_file", {"path": "/workspace/test_repo/file.txt"})

        assert res == (
            "error",
            'ERROR - new_str not found in tool params: (path="/workspace/test_repo/file.txt")',
        )

    def test_edit_file_str_replace(self):
        res = execute_tool_call(
            "edit_file",
            {
                "path": "/workspace/test_repo/file.txt",
                "old_str": "\nHello World",
                "new_str": "Goodbye",
            },
        )

        assert res == ("success", "File successfully updated")
        self.assert_file_content("file.txt", "old content\nGoodbye")

    def test_edit_file_path_not_in_sandbox(self):
        res = execute_tool_call(
            "edit_file",
            {"path": "/workspace/../llama_agent/main.py", "new_str": "new content"},
        )

        assert res == (
            "error",
            f"ERROR - File {SANDBOX_DIR}/../llama_agent/main.py does not exist",
        )

    def test_edit_file_path_not_symlink(self):
        # Create a symlink to a file outside of the sandbox. E.g., simulate to trying to steal credentials
        os.symlink(
            os.path.join(REPO_DIR, ".env.example"), os.path.join(self.test_dir, ".env")
        )

        res = execute_tool_call(
            "edit_file", {"path": "/workspace/test_repo/.env", "new_str": "new content"}
        )

        assert res == (
            "error",
            "ERROR - File /workspace/test_repo/.env is a symlink. Simlinks not allowed",
        )

    def test_view_file_path_not_in_sandbox(self):
        res = execute_tool_call(
            "view_file", {"path": "/workspace/../llama_agent/main.py"}
        )

        assert res == (
            "error",
            f"ERROR - File {SANDBOX_DIR}/../llama_agent/main.py does not exist",
        )

    def test_view_file_symlink(self):
        os.symlink(
            os.path.join(REPO_DIR, ".env.example"), os.path.join(self.test_dir, "pwned")
        )

        res = execute_tool_call("view_file", {"path": "/workspace/test_repo/pwned"})

        assert res == (
            "error",
            "ERROR - File /workspace/test_repo/pwned is a symlink. Simlinks not allowed",
        )

    def test_view_file_path_not_exists(self):
        res = execute_tool_call(
            "view_file", {"path": "/workspace/test_repo/does_not_exist"}
        )

        assert res == (
            "error",
            "ERROR - File /workspace/test_repo/does_not_exist does not exist. Please ensure the path is an absolute path and that the file exists.",
        )

    def test_view_file_success(self):
        res = execute_tool_call("view_file", {"path": "/workspace/test_repo/file.txt"})

        assert res == ("success", "old content\n\nHello World")
    
    def assert_file_content(self, path: str, expected_content: str) -> None:
        with open(os.path.join(self.test_dir, path), "r") as f:
            assert f.read() == expected_content


def add_to_git(dir: str) -> None:
    run(
        f"cd {dir} && git init && git add . && git commit -m 'Initial commit'",
        shell=True,
        check=True,
        capture_output=True,
    )
