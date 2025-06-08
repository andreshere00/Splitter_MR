import re
import sys


def convert_github_admonitions(md_text):
    lines = md_text.splitlines()
    out_lines = []
    i = 0
    while i < len(lines):
        match = re.match(r"^\s*>\s*\[!(\w+)\]", lines[i])
        if match:
            typ = match.group(1).lower()
            # Write the admonition header
            out_lines.append(f"!!! {typ}")
            i += 1
            # Collect and convert quoted lines
            while i < len(lines) and re.match(r"^\s*>", lines[i]):
                # Remove the leading '> ' and indent with 4 spaces
                out_lines.append("    " + re.sub(r"^\s*> ?", "", lines[i]))
                i += 1
            # Add an empty line for MkDocs rendering (optional)
            out_lines.append("")
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines)


if __name__ == "__main__":
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(in_path) as fin:
        md = fin.read()
    md2 = convert_github_admonitions(md)
    with open(out_path, "w") as fout:
        fout.write(md2)
