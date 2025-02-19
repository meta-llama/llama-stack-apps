import pytest
from llama_agent.github import Issue

class TestIssue:
    def test_basic_url(self):
        url = "https://github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
        issue = Issue(url)
        assert issue.owner == "aidando73"
        assert issue.repo == "bitbucket-syntax-highlighting"
        assert issue.issue_number == 67

    def test_basic_url_without_https(self):
        url = "github.com/aidando73/bitbucket-syntax-highlighting/issues/67"
        issue = Issue(url)
        assert issue.owner == "aidando73"
        assert issue.repo == "bitbucket-syntax-highlighting"
        assert issue.issue_number == 67

    def test_invalid_url(self):
        with pytest.raises(ValueError, match="Expected github.com as the domain"):
            Issue("https://not-github.com/owner/repo/issues/1")

    def test_invalid_url_no_issue(self):
        with pytest.raises(ValueError, match="Invalid GitHub issue URL format"):
            Issue("https://github.com/owner/repo")

    def test_invalid_url_no_issue_number(self):
        with pytest.raises(ValueError, match="Expected an issue number in the URL"):
            Issue("https://github.com/owner/repo/issues/")

    def test_issue_number_is_not_integer(self):
        with pytest.raises(ValueError, match="Expected an integer issue number"):
            Issue("https://github.com/owner/repo/issues/not_an_integer")
