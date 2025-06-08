import pandas as pd
import pytest

from splitter_mr.schema import ReaderOutput
from splitter_mr.splitter.splitters.row_column_splitter import RowColumnSplitter

# Helpers


def make_reader_output(text, method):
    return ReaderOutput(
        text=text,
        document_name="test",
        document_path="test",
        conversion_method=method,
        document_id="doc1",
        ocr_method=None,
        metadata={},
    )


def get_markdown_rows(df):
    """Returns a list of markdown rows (strings) for each data row in the dataframe."""
    rows = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        md = row.to_markdown(index=False).splitlines()
        rows.append(md[-1])
    return rows


def get_markdown_header(df):
    """Gets the header from dataframe in markdown foramt"""
    lines = df.head(0).to_markdown(index=False).splitlines()
    return "\n".join(lines[:2])


def parse_data_rows_from_markdown(markdown_chunk):
    """Return list of tuples representing the data rows in a markdown table chunk."""
    lines = markdown_chunk.strip().splitlines()
    # Data always starts after the first two lines (header and separator)
    data_lines = lines[2:]
    rows = []
    for line in data_lines:
        # Split on pipe, strip whitespace, skip empty
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and any(cells):
            rows.append(tuple(cells))
    return rows


def is_markdown_header_line(line, required_cols):
    """Check if a line looks like a markdown table header with all expected columns."""
    # Remove pipes and spaces, lowercase for robustness
    cols = [c.strip().lower() for c in line.strip("|").split("|")]
    return all(col in cols for col in required_cols)


# Test cases


@pytest.mark.parametrize(
    "num_rows, num_cols, overlap, expected_chunks",
    [
        (2, 0, 0, 2),  # 2 rows per chunk, no overlap
        (1, 0, 1, 4),  # 1 row per chunk, 1 overlap (should give 4 chunks for 4 rows)
        (0, 2, 1, 4),  # 2 cols per chunk, 1 col overlap
    ],
)
def test_json_tabular(num_rows, num_cols, overlap, expected_chunks):
    json_tabular = """
    [
        {"id": 1, "name": "A", "amount": 1, "Remark": "x"},
        {"id": 2, "name": "B", "amount": 2, "Remark": "y"},
        {"id": 3, "name": "C", "amount": 3, "Remark": "z"},
        {"id": 4, "name": "D", "amount": 4, "Remark": "w"}
    ]"""
    reader_output = make_reader_output(json_tabular, "json")
    splitter = RowColumnSplitter(
        num_rows=num_rows, num_cols=num_cols, chunk_overlap=overlap
    )
    output = splitter.split(reader_output)
    assert len(output.chunks) == expected_chunks


def test_markdown_table():
    md_table = (
        "| id | name | amount | Remark |\n"
        "| --- | --- | --- | --- |\n"
        "| 1 | Alice | 1 | ok |\n"
        "| 2 | Bob | 2 | good |\n"
        "| 3 | Carol | 3 | fair |\n"
        "| 4 | Dan | 4 | bad |"
    )
    reader_output = make_reader_output(md_table, "markdown")
    splitter = RowColumnSplitter(num_rows=2)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_csv_split():
    csv_content = (
        "id,name,amount,Remark\n"
        "1,Alice,1,ok\n"
        "2,Bob,2,good\n"
        "3,Carol,3,fair\n"
        "4,Dan,4,bad\n"
    )
    reader_output = make_reader_output(csv_content, "csv")
    splitter = RowColumnSplitter(num_rows=2)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_tsv_split():
    tsv_content = (
        "id\tname\tamount\tRemark\n"
        "1\tAlice\t1\tok\n"
        "2\tBob\t2\tgood\n"
        "3\tCarol\t3\tfair\n"
        "4\tDan\t4\tbad\n"
    )
    reader_output = make_reader_output(tsv_content, "tsv")
    splitter = RowColumnSplitter(num_rows=3)
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2  # first 3, last 1 row
    header = output.chunks[0].splitlines()[0]
    for col in ["id", "name", "amount", "Remark"]:
        assert col in header


def test_chunk_size_too_small():
    md_table = "| id | name |\n" "|----|------|\n" "| 1  | A    |\n"
    splitter = RowColumnSplitter(chunk_size=10, num_rows=0, num_cols=0)
    reader_output = make_reader_output(md_table, "markdown")
    with pytest.raises(
        ValueError, match="chunk_size is too small to fit header and at least one row."
    ):
        splitter.split(reader_output)


def test_chunk_size_only():
    md_table = (
        "| id | name |\n"
        "|----|------|\n"
        "| 1  | A    |\n"
        "| 2  | B    |\n"
        "| 3  | C    |\n"
        "| 4  | D    |\n"
    )
    splitter = RowColumnSplitter(chunk_size=60, num_rows=0, num_cols=0)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    for chunk in output.chunks:
        header_line = chunk.strip().splitlines()[0]
        assert is_markdown_header_line(header_line, ["id", "name"])
        assert len(chunk) <= 60
    # Check data rows as before
    found_rows = set()
    for chunk in output.chunks:
        for row in parse_data_rows_from_markdown(chunk):
            found_rows.add(row)
    expected = [("1", "A"), ("2", "B"), ("3", "C"), ("4", "D")]
    for row in expected:
        assert row in found_rows


def test_chunk_size_with_overlap():
    md_table = (
        "| id | name |\n"
        "|----|------|\n"
        "| 1  | A    |\n"
        "| 2  | B    |\n"
        "| 3  | C    |\n"
    )
    splitter = RowColumnSplitter(
        chunk_size=80, chunk_overlap=10, num_rows=0, num_cols=0
    )
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    for chunk in output.chunks:
        print("\n" + str(chunk) + "\n")
        header_line = chunk.strip().splitlines()[0]
        assert is_markdown_header_line(header_line, ["id", "name"])
        assert len(chunk) <= 80
    # Parse data rows from each chunk
    all_rows = [parse_data_rows_from_markdown(chunk) for chunk in output.chunks]
    # Check overlap: last row of chunk i == first row of chunk i+1
    for i in range(len(all_rows) - 1):
        assert all_rows[i][-1] == all_rows[i + 1][0]
    # All expected data rows are present
    expected = [("1", "A"), ("2", "B"), ("3", "C")]
    found = [row for rows in all_rows for row in rows]
    for row in expected:
        assert row in found


def test_empty_input():
    splitter = RowColumnSplitter(num_rows=2)
    reader_output = make_reader_output("", "markdown")
    output = splitter.split(reader_output)
    # Should return one chunk, possibly empty
    assert len(output.chunks) == 0
    assert output.chunks == []


def test_missing_headers():
    # No header line at all
    md_table = "| 1 | A |\n" "| 2 | B |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    # Pandas will treat first row as header
    assert "1" in output.chunks[0]
    assert "A" in output.chunks[0]


def test_malformed_table():
    md_table = (
        "| id | name |\n"
        "|----|------|\n"
        "| 1  | A |\n"  # Too few columns
        "2 | B |\n"  # Missing leading pipe
        "| 3 | C | X |\n"  # Too many columns
    )
    splitter = RowColumnSplitter(num_rows=2)
    reader_output = make_reader_output(md_table, "markdown")
    with pytest.raises(pd.errors.ParserError, match="Malformed markdown table"):
        splitter.split(reader_output)


def test_single_row():
    md_table = "| id | name |\n" "|----|------|\n" "| 1  | A    |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    assert len(output.chunks) == 1
    for col in ["id", "name", "A"]:
        assert col in output.chunks[0]


def test_one_column():
    md_table = "| id |\n" "|----|\n" "| 1  |\n" "| 2  |\n"
    splitter = RowColumnSplitter(num_rows=1)
    reader_output = make_reader_output(md_table, "markdown")
    output = splitter.split(reader_output)
    assert len(output.chunks) == 2
    for chunk in output.chunks:
        assert "id" in chunk
