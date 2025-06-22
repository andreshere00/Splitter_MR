import copy
from typing import List, Optional

from bs4 import BeautifulSoup

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class HTMLTagSplitter(BaseSplitter):
    """
    HTMLTagSplitter splits HTML content based on a specified tag.
    In case that this tag is not specified, it is automatically detected as
    the most frequent and shallowest tag.

    Args:
        chunk_size (int): maximum chunk size, in characters
        tag (str): lowest level of hierarchy where do you want to split the text.
    """

    def __init__(self, chunk_size: int = 10000, tag: Optional[str] = None):
        # TODO: chunk_size it is not necessary for this Splitter. Remove from BaseSplitter class.
        super().__init__(chunk_size)
        self.tag = tag

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits HTML in `reader_output['text']` using the specified tag or, if not specified,
        automatically selects the most frequent and shallowest tag.

        Args:
            reader_output (Dict[str, Any]): Dictionary containing at least a 'text' key
                (str) and optional document metadata (e.g., 'document_name', 'document_path').

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If `reader_output` does not contain a 'text' key or if the HTML cannot be parsed.

        Example:
            ```python
            from splitter_mr.splitter import HTMLTagSplitter

            # This dictionary has been obtained as the output from a Reader object.
            reader_output = ReaderOutput(
                text: "<html><body><div>Chunk 1</div><div>Chunk 2</div></body></html>",
                document_name: "example.html",
                document_path: "/path/to/example.html"
            )
            splitter = HTMLTagSplitter(tag="div")
            output = splitter.split(reader_output)
            print(output["chunks"])
            ```
            ```python
            [
            '<html><body><div>Chunk 1</div></body></html>',
            '<html><body><div>Chunk 2</div></body></html>'
            ]
            ```
        """
        html = reader_output.text
        soup = BeautifulSoup(html, "html.parser")
        tag = self.tag or self._auto_tag(soup)

        elements = soup.find_all(tag)

        chunks: List[str] = []
        buffer = []

        def get_table_header(el):
            """
            If the element or any of its parents is a <table>,
            extract the <thead> or first <tr> inside <table> as the header.
            Return a list of header elements (can be empty).
            """
            # Find the closest table ancestor (or self if it's a table)
            table = el.find_parent("table")
            if el.name == "table":
                table = el

            if not table:
                return []

            # Try to get <thead>
            thead = table.find("thead")
            if thead:
                return [copy.deepcopy(thead)]

            # Else fallback: first <tr> in table (usually header row)
            first_tr = table.find("tr")
            if first_tr:
                # Wrap it in a <thead> tag for consistency
                soup_tmp = BeautifulSoup("", "html.parser")
                thead_tag = soup_tmp.new_tag("thead")
                thead_tag.append(copy.deepcopy(first_tr))
                return [thead_tag]

            return []

        def build_chunk_html(elements):
            chunk_soup = BeautifulSoup("", "html.parser")
            html_tag = chunk_soup.new_tag("html")
            body_tag = chunk_soup.new_tag("body")
            html_tag.append(body_tag)
            chunk_soup.append(html_tag)

            # If the first element is inside a table, prepend the header rows once
            header_elems = []
            if elements:
                header_elems = get_table_header(elements[0])
                for he in header_elems:
                    body_tag.append(he)

            for el in elements:
                # Only append <tr> if it's NOT a duplicate of the header row
                if el.name == "tr":
                    # Check if all children are <th>
                    if all(
                        child.name == "th"
                        for child in el.children
                        if getattr(child, "name", None)
                    ):
                        # This <tr> is a header row (skip it if header already added)
                        if header_elems:
                            continue  # skip, already included in <thead>
                body_tag.append(copy.deepcopy(el))
            return str(chunk_soup)

        for el in elements:
            test_buffer = buffer + [el]
            test_chunk_str = build_chunk_html(test_buffer)
            if len(test_chunk_str) > self.chunk_size and buffer:
                chunk_str = build_chunk_html(buffer)
                chunks.append(chunk_str)
                buffer = [el]
            else:
                buffer.append(el)

        if buffer:
            chunk_str = build_chunk_html(buffer)
            chunks.append(chunk_str)

        chunk_ids = self._generate_chunk_ids(len(chunks))
        metadata = self._default_metadata()

        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="html_tag_splitter",
            split_params={"chunk_size": self.chunk_size, "tag": tag},
            metadata=metadata,
        )
        return output

    def _auto_tag(self, soup: BeautifulSoup) -> str:
        """
        Auto-detect the most repeated tag with the highest (shallowest) level of hierarchy.
        If no repeated tags are found, return the first tag found in <body> or fallback to 'div'.
        """
        from collections import Counter, defaultdict

        body = soup.find("body")
        if not body:
            return "div"

        # Traverse all tags in body, tracking tag: (count, min_depth)
        tag_counter = Counter()
        tag_min_depth = defaultdict(lambda: float("inf"))

        def traverse(el, depth=0):
            for child in el.children:
                if getattr(child, "name", None):
                    tag_counter[child.name] += 1
                    tag_min_depth[child.name] = min(tag_min_depth[child.name], depth)
                    traverse(child, depth + 1)

        traverse(body)

        if not tag_counter:
            # fallback to first tag
            for tag in body.find_all(True, recursive=True):
                return tag.name
            return "div"

        # Find tags with the maximum count
        max_count = max(tag_counter.values())
        candidates = [tag for tag, cnt in tag_counter.items() if cnt == max_count]
        # Of the most frequent, pick the one with the minimum depth (shallowest)
        tag = min(candidates, key=lambda tag: tag_min_depth[tag])
        return tag
