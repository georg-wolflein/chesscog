#!/bin/bash

# Script to generate HTML and PDF documentation

SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH/..

mkdir -p docs
poetry run pdoc3 --html --force --output-dir docs/html chesscog

# poetry run pdoc3 --pdf --config='docformat="google"' chesscog > docs.md
# # fix headings
# awk '{gsub(/-----=/,":\n")}1' docs.md > tmp.md && mv tmp.md docs.md
# pandoc  --metadata=title:"Documentation of the chesscog package" \
#         --from=markdown+abbreviations+tex_math_single_backslash  \
#         --pdf-engine=xelatex \
#         --toc --toc-depth=4 --output=docs/chesscog.pdf  docs.md
# rm docs.md