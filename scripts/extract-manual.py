# /// script
# requires-python = ">=3.11"
# dependencies = ["pypdf>=4"]
# ///
"""Turn your own copy of the OP-1 field user guide PDF into the text file that search_manual reads.

Teenage Engineering's guide is not distributed with op-bridge. Download the PDF from
https://teenage.engineering/guides/op-1 ("download as PDF"), then:

    uv run scripts/extract-manual.py ~/Downloads/op-1-field-user-guide.pdf

The text lands in docs/reference/op-1-field-user-guide-fw1.7.txt with a "===== PAGE n =====" line
before each page, which is how search_manual reports page numbers.
"""
import logging
import os
import sys

from pypdf import PdfReader

logging.getLogger("pypdf").setLevel(logging.ERROR)  # font-encoding notices; the extracted text is unaffected

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "reference", "op-1-field-user-guide-fw1.7.txt")


def main() -> int:
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        return 2
    out = sys.argv[2] if len(sys.argv) == 3 else OUT
    pages = PdfReader(sys.argv[1]).pages
    with open(out, "w") as f:
        for n, page in enumerate(pages, 1):
            f.write(f"\n\n===== PAGE {n} =====\n{page.extract_text() or ''}")
    print(f"wrote {len(pages)} pages to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
