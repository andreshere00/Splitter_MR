import html
import re

from bs4 import BeautifulSoup


class HtmlToMarkdown:
    """
    Converts HTML text to Markdown text, handling block and inline elements
    with proper formatting and whitespace control.

    Attributes:
        None
    """

    def __init__(self):
        pass

    def convert(self, html_text: str) -> str:
        """Convert HTML text to Markdown.

        Args:
            html_text (str): The HTML string to convert.

        Returns:
            str: The resulting Markdown string.
        """
        soup = BeautifulSoup(html_text, "html.parser")
        md = self._to_markdown(soup)
        # Remove multiple consecutive blank lines
        md = re.sub(r"\n{3,}", "\n\n", md)
        return md.strip()

    def _to_markdown(self, el, in_list=False, list_level=0, inline=False):
        """Recursive helper to convert HTML element to Markdown.

        Args:
            el: BeautifulSoup element or string.
            in_list (bool): True if inside a list context.
            list_level (int): Current list indentation level.
            inline (bool): True if within an inline context.

        Returns:
            str: Markdown-formatted string for the given element.
        """
        # Text node
        if isinstance(el, str):
            # Collapse whitespace in inline mode
            return html.unescape(el) if inline else html.unescape(el).strip()

        out = []
        for node in el.children if hasattr(el, "children") else []:
            name = getattr(node, "name", None)

            # ---- Block-level ----
            if name and re.match(r"h([1-6])", name):
                level = int(name[1])
                content = self._to_markdown(node, inline=True).strip()
                out.append(f"{'#' * level} {content}\n")
            elif name == "p":
                content = self._to_markdown(node, inline=True).strip()
                if content:
                    out.append(f"{content}\n")
            elif name == "blockquote":
                content = self._to_markdown(node, inline=True).strip().splitlines()
                out.append("\n".join([f"> {line}" for line in content]) + "\n")
            elif name == "ul":
                for li in node.find_all("li", recursive=False):
                    body = self._to_markdown(
                        li, in_list=True, list_level=list_level + 1, inline=True
                    )
                    indent = "  " * (list_level)
                    out.append(f"{indent}- {body.strip()}")
                out.append("")
            elif name == "ol":
                for idx, li in enumerate(node.find_all("li", recursive=False), 1):
                    body = self._to_markdown(
                        li, in_list=True, list_level=list_level + 1, inline=True
                    )
                    indent = "  " * (list_level)
                    out.append(f"{indent}{idx}. {body.strip()}")
                out.append("")
            elif name == "li":
                # Inline content inside list item, keep children together
                md = "".join(
                    [self._to_markdown(child, inline=True) for child in node.children]
                ).strip()
                out.append(md)
            elif name == "hr":
                out.append("---\n")
            elif name == "pre":
                code = node.code or node
                lang = code.get("class", [None])
                lang_name = ""
                for item in lang:
                    m = re.match(r"language-(\w+)|lang-(\w+)", item or "")
                    if m:
                        lang_name = m.group(1) or m.group(2)
                code_text = code.get_text() if code else node.get_text()
                code_text = html.unescape(code_text.rstrip("\n"))
                out.append(f"```{lang_name}\n{code_text}\n```\n")
            elif name == "table":
                md = self._table_to_md(node)
                out.append(md)
            elif name in ("div", "body", "html", "section", "article"):
                # Flatten children, treat as block
                out.append(self._to_markdown(node))
            # ---- Inline-level ----
            elif name in ("b", "strong"):
                content = self._to_markdown(node, inline=True)
                out.append(f"**{content}**")
            elif name in ("i", "em"):
                content = self._to_markdown(node, inline=True)
                out.append(f"*{content}*")
            elif name == "del":
                content = self._to_markdown(node, inline=True)
                out.append(f"~~{content}~~")
            elif name == "code" and node.parent.name != "pre":
                content = self._to_markdown(node, inline=True)
                out.append(f"`{content}`")
            elif name == "a":
                href = node.get("href", "")
                txt = self._to_markdown(node, inline=True)
                out.append(f"[{txt}]({href})")
            elif name == "img":
                alt = node.get("alt", "")
                src = node.get("src", "")
                out.append(f"![{alt}]({src})")
            # ---- Unknown tag or fallback ----
            else:
                if hasattr(node, "children"):
                    out.append(self._to_markdown(node, inline=inline))
                else:
                    txt = getattr(node, "string", "")
                    if txt:
                        out.append(
                            html.unescape(txt) if inline else html.unescape(txt).strip()
                        )
        # Remove extraneous newlines inside inline content
        joined = ""
        if inline:
            joined = "".join(out)
        else:
            joined = "\n".join(x for x in out if x is not None and x.strip())
        return joined

    def _table_to_md(self, table):
        """Convert HTML table to Markdown table.

        Args:
            table: BeautifulSoup table tag.

        Returns:
            str: Markdown table representation.
        """
        rows = []
        for tr in table.find_all("tr"):
            row = []
            for td in tr.find_all(["td", "th"]):
                row.append(self._to_markdown(td, inline=True).strip())
            rows.append(row)
        if not rows:
            return ""
        # Markdown table: header, separator, data
        header = "| " + " | ".join(rows[0]) + " |"
        sep = "| " + " | ".join("---" for _ in rows[0]) + " |"
        lines = [header, sep]
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines) + "\n"
