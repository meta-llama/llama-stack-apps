import pytest
from llama_agent.main import main

class TestApp:
    @pytest.fixture(autouse=True)
    def setup_method(self, monkeypatch):
        """Set up test environment before each test method"""
        # Ensure environment variables are set for tests
        monkeypatch.setenv("GITHUB_API_KEY", "test_key")
        monkeypatch.setenv("LLAMA_STACK_URL", "http://localhost:5000")

    def test_no_github_api_key(self, monkeypatch):
        monkeypatch.delenv("GITHUB_API_KEY", raising=False)

        with pytest.raises(
            ValueError, match="GITHUB_API_KEY is not set in the environment variables"
        ):
            main(
                issue_url="https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
            )

    def test_no_llama_stack_url(self, monkeypatch):
        monkeypatch.delenv("LLAMA_STACK_URL", raising=False)

        with pytest.raises(
            ValueError, match="LLAMA_STACK_URL is not set in the environment variables"
        ):
            main(
                issue_url="https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
            )