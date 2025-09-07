# md2ipynb.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

FENCE = "```"


def _flush_markdown(cells: List[nbformat.NotebookNode], buf: List[str]) -> None:
    text = "".join(buf).rstrip("\n")
    if text.strip():
        cells.append(new_markdown_cell(text))


def _flush_code(
    cells: List[nbformat.NotebookNode],
    buf: List[str],
    lang: Optional[str],
) -> None:
    code = "".join(buf).rstrip("\n")
    if code.strip():
        cell = new_code_cell(code)
        if lang:
            # Store language hint without forcing kernel change
            cell.metadata["language"] = lang
        cells.append(cell)


def convert_md_to_ipynb(
    md_path: str | Path,
    ipynb_path: str | Path,
    *,
    default_language: str = "python",
) -> Dict[str, Any]:
    """
    Convert a Markdown file to a Jupyter notebook.

    - Fenced code blocks (```lang ... ```) become code cells (language stored in cell.metadata.language)
    - Everything else becomes markdown cells
    - Unclosed fences at EOF are treated as code

    Returns a summary dict.
    """
    mdp = Path(md_path)
    nbp = Path(ipynb_path)

    if not mdp.exists():
        raise FileNotFoundError(f"Markdown file not found: {mdp}")

    # Ensure parent directory exists for output
    nbp.parent.mkdir(parents=True, exist_ok=True)

    with mdp.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    cells: List[nbformat.NotebookNode] = []
    in_code = False
    code_lang: Optional[str] = None
    buffer: List[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect fenced code block start or end (``` possibly with language after)
        if not in_code and line.startswith(FENCE):
            in_code = True
            # e.g., "```python\n" -> "python"
            code_lang = line.strip().lstrip("`").strip() or None
            _flush_markdown(cells, buffer)
            buffer = []
        elif in_code and line.startswith(FENCE):
            in_code = False
            _flush_code(cells, buffer, code_lang)
            buffer = []
            code_lang = None
        else:
            buffer.append(line)
        i += 1

    # Flush any remaining buffer
    if in_code:
        _flush_code(cells, buffer, code_lang)
    else:
        _flush_markdown(cells, buffer)

    nb = new_notebook(
        cells=cells,
        metadata={
            "language_info": {"name": default_language},
            "orig_format": "markdown",
            "source_file": str(mdp),
        },
    )

    with nbp.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    summary = {
        "source_markdown": str(mdp),
        "output_notebook": str(nbp),
        "total_cells": len(cells),
        "code_cells": sum(1 for c in cells if c.cell_type == "code"),
        "markdown_cells": sum(1 for c in cells if c.cell_type == "markdown"),
    }
    return summary


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a Markdown file to a Jupyter notebook with fenced code blocks as code cells."
    )
    p.add_argument(
        "--md",
        "--md-path",
        dest="md_path",
        required=True,
        help="Path to the source Markdown file.",
    )
    p.add_argument(
        "--ipynb",
        "--ipynb-path",
        dest="ipynb_path",
        required=True,
        help="Path to the output .ipynb file.",
    )
    p.add_argument(
        "--default-language",
        dest="default_language",
        default="python",
        help="Default language to record in notebook metadata (does not change kernel).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        summary = convert_md_to_ipynb(
            args.md_path,
            args.ipynb_path,
            default_language=args.default_language,
        )
        # Print a compact JSON-like summary for CLI use
        print(
            f"Converted: {summary['source_markdown']} -> {summary['output_notebook']} "
            f"(cells: {summary['total_cells']}, "
            f"code: {summary['code_cells']}, md: {summary['markdown_cells']})"
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
