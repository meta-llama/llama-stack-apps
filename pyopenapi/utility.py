import importlib.resources
import json
import sys
import typing
from typing import TextIO

from strong_typing.schema import StrictJsonType, object_to_json

from .generator import Generator
from .options import Options
from .specification import Document


class Specification:
    document: Document

    def __init__(self, endpoint: type, options: Options):
        generator = Generator(endpoint, options)
        self.document = generator.generate()

    def get_json(self) -> StrictJsonType:
        """
        Returns the OpenAPI specification as a Python data type (e.g. `dict` for an object, `list` for an array).

        The result can be serialized to a JSON string with `json.dump` or `json.dumps`.
        """

        json_doc = typing.cast(StrictJsonType, object_to_json(self.document))

        if isinstance(json_doc, dict):
            # rename vendor-specific properties
            tag_groups = json_doc.pop("tagGroups", None)
            if tag_groups:
                json_doc["x-tagGroups"] = tag_groups
            tags = json_doc.get("tags")
            if tags and isinstance(tags, list):
                for tag in tags:
                    if not isinstance(tag, dict):
                        continue

                    display_name = tag.pop("displayName", None)
                    if display_name:
                        tag["x-displayName"] = display_name

        return json_doc

    def get_json_string(self, pretty_print: bool = False) -> str:
        """
        Returns the OpenAPI specification as a JSON string.

        :param pretty_print: Whether to use line indents to beautify the output.
        """

        json_doc = self.get_json()
        if pretty_print:
            return json.dumps(
                json_doc, check_circular=False, ensure_ascii=False, indent=4
            )
        else:
            return json.dumps(
                json_doc,
                check_circular=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def write_json(self, f: TextIO, pretty_print: bool = False) -> None:
        """
        Writes the OpenAPI specification to a file as a JSON string.

        :param pretty_print: Whether to use line indents to beautify the output.
        """

        json_doc = self.get_json()
        if pretty_print:
            json.dump(
                json_doc,
                f,
                check_circular=False,
                ensure_ascii=False,
                indent=4,
            )
        else:
            json.dump(
                json_doc,
                f,
                check_circular=False,
                ensure_ascii=False,
                separators=(",", ":"),
            )

    def write_html(self, f: TextIO, pretty_print: bool = False) -> None:
        """
        Creates a stand-alone HTML page for the OpenAPI specification with ReDoc.

        :param pretty_print: Whether to use line indents to beautify the JSON string in the HTML file.
        """

        if sys.version_info >= (3, 9):
            with importlib.resources.files(__package__).joinpath("template.html").open(
                encoding="utf-8", errors="strict"
            ) as html_template_file:
                html_template = html_template_file.read()
        else:
            with importlib.resources.open_text(
                __package__, "template.html", encoding="utf-8", errors="strict"
            ) as html_template_file:
                html_template = html_template_file.read()

        html = html_template.replace(
            "{ /* OPENAPI_SPECIFICATION */ }",
            self.get_json_string(pretty_print=pretty_print),
        )

        f.write(html)
