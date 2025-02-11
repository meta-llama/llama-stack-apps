import os
from typing import List
from subprocess import run

class Directory:
    name: str
    files: set[str]
    directories: set["Directory"]

    def __init__(self, name: str):
        self.name = name
        self.files = set()
        self.directories = set()

    def add_file(self, file: str):
        self.files.add(file)

    def add_directory(self, directory: "Directory"):
        self.directories.add(directory)

    def __str__(self):
        return f"{self.name} ({len(self.files)} files, {len(self.directories)} directories)"

    def __repr__(self):
        return f"{self.name} ({len(self.files)} files, {len(self.directories)} directories)"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


def list_files_in_repo(path: str, depth: int = 1) -> List[str]:
    """
    List all files in the given path, up to the given depth
    Returns a list of file paths, including directories
    Directories are represented by a trailing slash
    Directories are displayed first, then files

    Assumes the directory is in a git repo. Will exclude any files that are not in the git repo
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")

    if os.path.isfile(path):
        return [path]

    # We use git ls-tree to ignore any files like .git/
    cmd = run(
        f"cd {path} && git ls-tree -r --name-only HEAD",
        shell=True,
        text=True,
        capture_output=True,
    )
    files = cmd.stdout
    if cmd.returncode != 0:
        raise AssertionError(f"Failed to list files in repo: {cmd.stderr}")

    files = files.splitlines()

    root = Directory(path)
    for file in files:
        # Sometimes git ls-tree returns files with quotes around them
        # E.g., for files with spaces in their name
        file = file.strip('"')
        parts = file.split("/")
        cur = root
        for i in range(len(parts)):
            if i + 1 > depth:
                break
            if i == len(parts) - 1:
                cur.add_file(parts[i])
            else:
                if Directory(parts[i]) not in cur.directories:
                    temp = Directory(parts[i])
                    cur.add_directory(temp)
                    cur = temp
                else:
                    cur = next(d for d in cur.directories if d.name == parts[i])
    res = []

    def dfs(directory: Directory, path=""):
        # Recursively process subdirectories
        for subdir in sorted(directory.directories, key=lambda x: x.name):
            subdir_path = os.path.join(path, subdir.name)
            res.append(subdir_path + "/")  # Add trailing slash for directories
            dfs(subdir, subdir_path)

            # Add all files in current directory
        for file in sorted(directory.files):
            res.append(os.path.join(path, file))

    dfs(root)
    return res
