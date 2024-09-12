import os

import fire
import requests

from fpdf import FPDF
from lxml import etree


def get_github_issue(repo_name: str, output_path: str):
    """Enable github issue vectorizing"""
    base_url = f"https://github.com/{repo_name}/issues?q=is%3Aissue+is%3Aclosed"
    print("url is:", base_url)

    def fetch_issues(url):
        response = requests.get(url)
        html_content = response.content
        tree = etree.HTML(html_content)
        return tree

    def extract_issues(tree):
        issues = tree.xpath('//div[contains(@class, "Box-row")]')
        return issues

    def get_next_page(tree):
        next_page_elements = tree.xpath('//a[@class="next_page"]/@href')
        if next_page_elements:
            return f"https://github.com{next_page_elements[0]}"
        return None

    # Initialize variables
    current_url = base_url
    all_issues = []
    seen_urls = set()

    # Loop through all pages
    while current_url:
        tree = fetch_issues(current_url)
        issues = extract_issues(tree)
        for issue in issues:
            issue_url_elements = issue.xpath('.//a[@data-hovercard-type="issue"]/@href')
            issue_url = (
                f"https://github.com{issue_url_elements[0]}"
                if issue_url_elements
                else "No URL"
            )
            if issue_url not in seen_urls:
                seen_urls.add(issue_url)
                all_issues.append(issue)
        current_url = get_next_page(tree)

    # Prepare the Markdown content
    total_issues = len(all_issues)
    print("Total Issues: ", total_issues)
    md_content = f"# \n\nTotal Closed Issues: {total_issues}\n\n"

    for issue in all_issues:
        title_elements = issue.xpath('.//a[@data-hovercard-type="issue"]/text()')
        title = title_elements[0].strip() if title_elements else "No title"

        issue_url_elements = issue.xpath('.//a[@data-hovercard-type="issue"]/@href')
        issue_url = (
            f"https://github.com{issue_url_elements[0]}"
            if issue_url_elements
            else "No URL"
        )
        print(issue_url)
        response = requests.get(issue_url)
        page_source = response.text
        tree = etree.HTML(page_source)
        content = tree.xpath("//table//td")[0].xpath(
            "string(.)"
        )  # this is only description, not full content. May need update

        # Append to Markdown content
        md_content += f"##Issue: {title}\n\n"
        md_content += f"**Description:**\n\n{content}\n\n"
        md_content += f"**Solution:** {issue_url}\n\n"
        md_content += "-------------------------------------------------------------------------------------------\n\n"

    # Save the Markdown content to a file
    md_filename = f"{output_path}.md"
    with open(md_filename, "w") as md_file:
        md_file.write(md_content)

    print("The GitHub issues have been successfully written to", md_filename)

    # Function to convert Markdown to PDF
    def convert_md_to_pdf(md_filename, pdf_filename):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        with open(md_filename, "r", encoding="utf-8") as md_file:
            for line in md_file:
                pdf.multi_cell(
                    0, 10, line.encode("latin-1", "replace").decode("latin-1")
                )

        pdf.output(pdf_filename)
        print(f"The Markdown file has been successfully converted to {pdf_filename}")

    # Convert the .md file to a .pdf file
    if os.path.isfile(output_path):
        os.remove(output_path)
    convert_md_to_pdf(md_filename, output_path)


if __name__ == "__main__":
    fire.Fire(get_github_issue)
