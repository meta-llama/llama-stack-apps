class Issue:
    repo: str
    owner: str
    issue_number: int

    def __init__(self, url: str):
        """
        Parse the issue url into the owner, repo and issue number
        """
        # Truncate the https:// prefix if it exists
        base_url = url
        if url.startswith("https://"):
            url = url[len("https://") :]

        # Check that the domain is github.com
        if not url.startswith("github.com"):
            raise ValueError(f"Expected github.com as the domain: {base_url}")

        parts = url.split("/")
        if (
            len(parts) < 5
        ):  # We expect 5 parts: github.com/owner/repo/issues/issue_number
            raise ValueError("Invalid GitHub issue URL format")

        self.owner = parts[1]
        self.repo = parts[2]
        if parts[3] != "issues":
            raise ValueError(f"Expected /issues/ in the URL: {base_url}")

        if parts[4] == "":
            raise ValueError(f"Expected an issue number in the URL: {base_url}")

        try:
            self.issue_number = int(parts[4])  # Issue number
        except ValueError:
            raise ValueError(f"Expected an integer issue number: {parts[4]}")
