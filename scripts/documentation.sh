#!/bin/bash
python3 scripts/md_admonition_convert.py README.md docs/index.md
cp CHANGELOG.md docs/CHANGELOG.md
cp -rf assets/ docs/assets/
