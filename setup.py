# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from setuptools import find_packages, setup


# Function to read the requirements.txt file
def read_requirements():
    with open("requirements.txt") as req:
        content = req.readlines()
    return [line.strip() for line in content]


setup(
    name="llama_agentic_system",
    version="0.0.2",
    author="Meta Llama",
    author_email="llama-oss@meta.com",
    description="Llama Agentic System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/meta-llama/llama-agentic-system",
    packages=find_packages(),
    classifiers=[],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    include_package_data=True,
)
