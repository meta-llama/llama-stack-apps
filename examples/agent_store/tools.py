import asyncio
import json
import os
import re
from typing import Dict, List

import aiohttp
import requests
from bs4 import BeautifulSoup

from dotenv import load_dotenv

from llama_models.llama3.api.datatypes import ToolParamDefinition
from llama_toolchain.tools.custom.datatypes import SingleMessageCustomTool


load_dotenv()


class SearchAndBrowse(SingleMessageCustomTool):
    """Tool to search web for relevant links, browse those pages
    and return information from those pages
    """

    def get_name(self) -> str:
        return "search_and_browse"

    def get_description(self) -> str:
        return "Search the web for relevant links and get content from those links"

    def get_params_definition(self) -> Dict[str, ToolParamDefinition]:
        return {
            "query": ToolParamDefinition(
                param_type="string", description="search query", required=True
            ),
        }

    async def run_impl(self, query: str) -> str:
        from llama_toolchain.tools.builtin import BingSearch

        api_key = os.getenv("BING_SEARCH_API_KEY")
        bing_search = BingSearch(api_key=api_key, top_k=2)
        search_results = await bing_search.search(query)
        results = json.loads(search_results)
        urls = []
        for res in results["top_k"]:
            if not isinstance(res, list):
                urls.append(res["url"])
            else:
                urls += [r["url"] for r in res]

        # res = await fetch_and_extract_all_urls(urls)
        return fetch_url_content(urls)


def fetch_html_content(url):
    try:
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Return the HTML content
        return response.text
    except requests.RequestException as e:
        return f"An error occurred: {e}"


def fetch_url_content(urls: List[str]) -> List[str]:
    results = {}
    for url in urls:
        results[url] = extract_main_text(fetch_html_content(url))[:2000]
    return results


async def fetch_url(session: aiohttp.ClientSession, url: str) -> str:
    async with session.get(url) as response:
        return await response.text()


def extract_main_text(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()

    # Get text
    text = soup.get_text()

    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())

    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))

    # Drop blank lines
    text = "\n".join(chunk for chunk in chunks if chunk)

    return text


async def fetch_and_extract_all_urls(urls: List[str]) -> Dict[str, str]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and extract main text
    processed_results = {}
    for url, result in zip(urls, results):
        if isinstance(result, str):
            processed_results[url] = extract_main_text(result)
        else:
            processed_results[url] = f"Error: {str(result)}"

    return processed_results
