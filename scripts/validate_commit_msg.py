#!/usr/bin/env python
import re
import sys


def main():
    # The commit message file is passed as the first argument.
    commit_msg_filepath = sys.argv[1]
    with open(commit_msg_filepath, "r", encoding="utf-8") as f:
        commit_msg = f.read().strip()

    # Regex to enforce the format:
    # <tipo>(<alcance>): <breve descripción del cambio>
    # Allowed types: fix, feat, docs, refactor, test, style, perf, build, ci, chore
    pattern = re.compile(
        r"^(fix|feat|docs|refactor|test|style|perf|build|ci|chore)"
        r"(\([a-zA-Z0-9_-]+\))?: .+"
    )

    if not pattern.match(commit_msg):
        print("ERROR: Commit message does not follow the required format:")
        print("<tipo>(<alcance>): <breve descripción del cambio>")
        print(
            "Allowed tipos: fix, feat, docs, refactor, test, style, "
            "perf, build, ci, chore"
        )
        sys.exit(1)

    else:
        print("✅ Commit message properly structured.")


if __name__ == "__main__":
    main()
