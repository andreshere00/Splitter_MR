import io
import json
import re
from typing import Any, Dict, Union

import pandas as pd

from ...schema import ReaderOutput, SplitterOutput
from ..base_splitter import BaseSplitter


class RowColumnSplitter(BaseSplitter):
    """
    RowColumnSplitter splits tabular data (such as CSV, TSV, Markdown tables, or JSON tables)
    into smaller tables based on rows, columns, or by total character size while preserving row integrity.

    This splitter supports several modes:

    - **By rows**: Split the table into chunks with a fixed number of rows, with optional overlapping
        rows between chunks.
    - **By columns**: Split the table into chunks by columns, with optional overlapping columns between chunks.
    - **By chunk size**: Split the table into markdown-formatted table chunks, where each chunk contains
        as many complete rows as fit under the specified character limit, optionally overlapping a fixed
        number of rows between chunks.

    This is useful for splitting large tabular files for downstream processing, LLM ingestion,
    or display, while preserving semantic and structural integrity of the data.

    Args:
        chunk_size (int): Maximum number of characters per chunk (when using character-based splitting).
        num_rows (int): Number of rows per chunk. Mutually exclusive with num_cols.
        num_cols (int): Number of columns per chunk. Mutually exclusive with num_rows.
        chunk_overlap (Union[int, float]): Number of overlapping rows or columns between chunks.
            If a float in (0,1), interpreted as a percentage of rows or columns. If integer, the number of
            overlapping rows/columns. When chunking by character size, this refers to the number of overlapping
            rows (not characters).

    Supported formats: CSV, TSV, TXT, Markdown table, JSON (tabular: list of dicts or dict of lists).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        num_rows: int = 0,
        num_cols: int = 0,
        chunk_overlap: Union[int, float] = 0,
    ):
        super().__init__(chunk_size)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.chunk_overlap = chunk_overlap

        if num_rows and num_cols:
            raise ValueError("num_rows and num_cols are mutually exclusive")
        if isinstance(chunk_overlap, float) and chunk_overlap >= 1:
            raise ValueError("chunk_overlap as float must be < 1")

    def split(self, reader_output: ReaderOutput) -> SplitterOutput:
        """
        Splits the input tabular data into multiple markdown table chunks according to the specified
        chunking strategy. Each output chunk is a complete markdown table with header, and will never
        cut a row in half. The overlap is always applied in terms of full rows or columns.

        Args:
            reader_output (Dict[str, Any]):
                Dictionary output from a Reader, containing at least:
                    - 'text': The tabular data as string.
                    - 'conversion_method': Format of the input ('csv', 'tsv', 'markdown', 'json', etc.).
                    - Additional document metadata fields (optional).

        Returns:
            SplitterOutput: Dataclass defining the output structure for all splitters.

        Raises:
            ValueError: If both num_rows and num_cols are set.
            ValueError: If chunk_overlap as float is not in [0,1).
            ValueError: If chunk_size is too small to fit the header and at least one data row.

        Example:
            ```python
            reader_output = ReaderOutput(
                text: '| id | name |\\n|----|------|\\n| 1  | A    |\\n| 2  | B    |\\n| 3  | C    |',
                conversion_method: "markdown",
                document_name: "table.md",
                document_path: "/path/table.md",
            )
            splitter = RowColumnSplitter(chunk_size=80, chunk_overlap=20)
            output = splitter.split(reader_output)
            for chunk in output["chunks"]:
                print("\\n" + str(chunk) + "\\n")
            ```
            ```python
            | id   | name   |
            |------|--------|
            |  1   | A      |
            |  2   | B      |

            | id   | name   |
            |------|--------|
            |  2   | B      |
            |  3   | C      |
            ```
        """
        # Step 1. Parse the table depending on conversion_method
        df = self._load_tabular(reader_output)
        orig_method = reader_output.conversion_method
        col_names = df.columns.tolist()

        # Step 2. Split logic
        chunks = []
        meta_per_chunk = []

        # If splitting strategy is by rows
        if self.num_rows > 0:
            overlap = self._get_overlap(self.num_rows)
            for i in range(
                0,
                len(df),
                self.num_rows - overlap if (self.num_rows - overlap) > 0 else 1,
            ):
                chunk_df = df.iloc[i : i + self.num_rows]
                if not chunk_df.empty:
                    chunk_str = self._to_str(chunk_df, orig_method)
                    chunks.append(chunk_str)
                    meta_per_chunk.append(
                        {"rows": chunk_df.index.tolist(), "type": "row"}
                    )
        # If splitting strategy is by columns
        elif self.num_cols > 0:
            overlap = self._get_overlap(self.num_cols)
            total_cols = len(col_names)
            for i in range(
                0,
                total_cols,
                self.num_cols - overlap if (self.num_cols - overlap) > 0 else 1,
            ):
                sel_cols = col_names[i : i + self.num_cols]
                if sel_cols:
                    chunk_df = df[sel_cols]
                    chunk_str = self._to_str(chunk_df, orig_method, colwise=True)
                    chunks.append(chunk_str)
                    meta_per_chunk.append({"cols": sel_cols, "type": "column"})
        # If splitting strategy is given by the chunk_size
        else:
            header_lines = self._get_markdown_header(df)
            header_length = len(header_lines)

            row_md_list = [self._get_markdown_row(df, i) for i in range(len(df))]
            row_len_list = [len(r) + 1 for r in row_md_list]  # +1 for newline

            if self.chunk_size < header_length + row_len_list[0]:
                raise ValueError(
                    "chunk_size is too small to fit header and at least one row."
                )

            # Compute overlapping and headers in markdown tables
            chunks = []
            meta_per_chunk = []
            i = 0
            n = len(row_md_list)
            overlap = self._get_overlap(1)
            while i < n:
                curr_chunk = []
                curr_len = header_length
                j = i
                while j < n and curr_len + row_len_list[j] <= self.chunk_size:
                    curr_chunk.append(row_md_list[j])
                    curr_len += row_len_list[j]
                    j += 1

                rows_in_chunk = j - i
                chunk_str = header_lines + "\n".join(curr_chunk)
                chunks.append(chunk_str)
                meta_per_chunk.append({"rows": list(range(i, j)), "type": "char_row"})

                # --- compute overlap AFTER we know rows_in_chunk ---
                if isinstance(self.chunk_overlap, float):
                    overlap_rows = int(rows_in_chunk * self.chunk_overlap)
                else:
                    overlap_rows = int(self.chunk_overlap)

                # make sure we don’t loop forever
                overlap_rows = min(overlap_rows, rows_in_chunk - 1)
                i = j - overlap_rows

        # Generate chunk_id
        chunk_ids = self._generate_chunk_ids(len(chunks))

        # Return output
        output = SplitterOutput(
            chunks=chunks,
            chunk_id=chunk_ids,
            document_name=reader_output.document_name,
            document_path=reader_output.document_path,
            document_id=reader_output.document_id,
            conversion_method=reader_output.conversion_method,
            reader_method=reader_output.reader_method,
            ocr_method=reader_output.ocr_method,
            split_method="row_column_splitter",
            split_params={
                "chunk_size": self.chunk_size,
                "num_rows": self.num_rows,
                "num_cols": self.num_cols,
                "chunk_overlap": self.chunk_overlap,
            },
            metadata=meta_per_chunk,
        )
        return output

    # Helper functions

    def _get_overlap(self, base: int):
        """
        Returns the overlap value as an integer, based on the configured chunk_overlap.

        If chunk_overlap is a float in (0,1), computes the overlap as a percentage of `base`.
        If chunk_overlap is an integer, returns it directly.

        Args:
            base (int): The base number (rows or columns) to compute the overlap from.
        Returns:
            int: The overlap as an integer.
        """
        if isinstance(self.chunk_overlap, float):
            return int(base * self.chunk_overlap)
        return int(self.chunk_overlap)

    def _load_tabular(self, reader_output: Dict[str, Any]) -> pd.DataFrame:
        """
        Loads and parses the input tabular data from a Reader output dictionary
        into a pandas DataFrame, based on its format.

        If the input is empty, returns an empty DataFrame.
        If the input is malformed (e.g., badly formatted markdown/CSV/TSV), a
        pandas.errors.ParserError is raised.

        Supports Markdown, CSV, TSV, TXT, and tabular JSON.

        Args:
            reader_output (Dict[str, Any]): Dictionary containing the text and conversion_method.

        Returns:
            pd.DataFrame: The loaded table as a DataFrame.

        Raises:
            pandas.errors.ParserError: If the input table is malformed and cannot be parsed.
        """
        text = reader_output.text
        # Return a void dataframe is a empty file is provided
        if not text or not text.strip():
            return pd.DataFrame()
        method = reader_output.conversion_method
        if method == "markdown":
            return self._parse_markdown_table(text)
        elif method == "csv" or method == "txt":
            return pd.read_csv(io.StringIO(text))
        elif method == "tsv":
            return pd.read_csv(io.StringIO(text), sep="\t")
        else:
            # Try JSON
            try:
                js = json.loads(text)
                if isinstance(js, list) and all(isinstance(row, dict) for row in js):
                    return pd.DataFrame(js)
                elif isinstance(js, dict):  # e.g., {col: [vals]}
                    return pd.DataFrame(js)
            except Exception:
                pass
            # Fallback: try CSV
            return pd.read_csv(io.StringIO(text))

    def _parse_markdown_table(self, md: str) -> pd.DataFrame:
        """
        Parses a markdown table string into a pandas DataFrame.

        Ignores non-table lines and trims markdown-specific formatting.
        Also handles the separator line (---) in the header.

        Args:
            md (str): The markdown table as a string.

        Returns:
            pd.DataFrame: Parsed table as a DataFrame.

        Raises:
            pandas.errors.ParserError: If the markdown table is malformed and cannot be parsed.
        """
        # Remove any lines not part of the table (e.g., text before/after)
        table_lines = []
        started = False
        for line in md.splitlines():
            if re.match(r"^\s*\|.*\|\s*$", line):
                started = True
                table_lines.append(line.strip())
            elif started and not line.strip():
                break  # stop at first blank line after table
        table_md = "\n".join(table_lines)
        table_io = io.StringIO(
            re.sub(
                r"^\s*\|",
                "",
                re.sub(r"\|\s*$", "", table_md, flags=re.MULTILINE),
                flags=re.MULTILINE,
            )
        )
        try:
            df = pd.read_csv(table_io, sep="|").rename(
                lambda x: x.strip(), axis="columns"
            )
        except pd.errors.ParserError as e:
            # Propagate the ParserError for your test to catch
            raise pd.errors.ParserError(f"Malformed markdown table: {e}") from e
        if not df.empty and all(re.match(r"^-+$", str(x).strip()) for x in df.iloc[0]):
            df = df.drop(df.index[0]).reset_index(drop=True)
        return df

    def _to_str(self, df: pd.DataFrame, method: str, colwise: bool = False) -> str:
        """
        Converts a DataFrame chunk to a string for output,
        either as a markdown table, CSV, or a list of columns.

        Args:
            df (pd.DataFrame): DataFrame chunk to convert.
            method (str): Input file format (for output style).
            colwise (bool): If True, output as a list of columns (used in column chunking).

        Returns:
            str: The chunk as a formatted string.
        """
        if colwise:
            # List of columns: output as a list of lists
            return (
                "["
                + ", ".join(  # noqa: W503
                    [str([col] + df[col].tolist()) for col in df.columns]  # noqa: W503
                )
                + "]"  # noqa: W503
            )
        if method == "markdown" or "md":
            # Use markdown table format
            return df.to_markdown(index=False)
        else:
            # Default to CSV format
            output = io.StringIO()
            df.to_csv(output, index=False)
            return output.getvalue().strip("\n")

    @staticmethod
    def _get_markdown_header(df):
        """
        Returns the header and separator lines for a markdown table as a string.

        Args:
            df (pd.DataFrame): DataFrame representing the table.

        Returns:
            str: Markdown table header and separator (with trailing newline).
        """

        lines = df.head(0).to_markdown(index=False).splitlines()
        return "\n".join(lines[:2]) + "\n"

    @staticmethod
    def _get_markdown_row(df, row_idx):
        """
        Returns a single row from the DataFrame formatted as a markdown table row.

        Args:
            df (pd.DataFrame): DataFrame containing the table.
            row_idx (int): Index of the row to extract.

        Returns:
            str: The markdown-formatted row string.
        """
        row = df.iloc[[row_idx]]
        # Get the full markdown output (with header),
        # extract only the last line (the data row)
        md = row.to_markdown(index=False).splitlines()
        return md[-1]
