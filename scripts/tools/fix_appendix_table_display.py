#!/usr/bin/env python3
"""Post-process the LaTeX appendix table for safer line wrapping and alignment."""

from __future__ import annotations

import re
from pathlib import Path

TABLE_PATH = Path("outputs/tables/appendix/appendix_phase1_full_statistics.tex")

# Matches either the original or the previously-fixed colspec.
COLSPEC_RE = re.compile(
    r"\\begin\{longtable\}\{[^}]*\}", re.DOTALL
)

NEW_COLSPEC = (
    "\\begin{longtable}{\n"
    "  >{\\raggedright\\arraybackslash}p{1.3cm}\n"
    "  >{\\raggedright\\arraybackslash}p{2.8cm}\n"
    "  >{\\raggedright\\arraybackslash}p{6cm}\n"
    "  >{\\raggedright\\arraybackslash}p{1cm}\n"
    "  >{\\raggedright\\arraybackslash}p{2.6cm}\n"
    "  >{\\raggedright\\arraybackslash}p{2.8cm}\n"
    "  >{\\raggedleft\\arraybackslash}p{0.7cm}\n"
    "  >{\\raggedleft\\arraybackslash}p{0.9cm}\n"
    "  >{\\raggedleft\\arraybackslash}p{0.8cm}\n"
    "  >{\\raggedleft\\arraybackslash}p{0.8cm}\n"
    "  >{\\raggedleft\\arraybackslash}p{1.3cm}\n"
    "}"
)

ARRAYSTRETCH_LINE = "\\renewcommand{\\arraystretch}{1.15}"

# Structural / header lines that must not be treated as data rows.
SKIP_PREFIXES = (
    r"\begin{longtable}",
    r"\caption{",
    r"\label{",
    r"\toprule",
    r"\midrule",
    r"\bottomrule",
    r"\endfirsthead",
    r"\endhead",
    r"\endfoot",
    r"\endlastfoot",
    r"\multicolumn",
    "Dataset & Family & Method",  # header row
    r"\end{longtable}",
    "  >{",  # continuation lines of a multi-line colspec
    "}",     # closing brace of a multi-line colspec
)


def add_breaks_to_first_three_columns(line: str) -> str:
    parts = line.split(" & ")
    if len(parts) < 11:
        return line  # not a standard 11-column data row

    for i in range(3):  # Dataset, Family, Method
        # Avoid double-inserting if the script is run twice.
        parts[i] = parts[i].replace(r"\_\allowbreak{}", r"\_")
        parts[i] = parts[i].replace(r"\_", r"\_\allowbreak{}")

    return " & ".join(parts)


def main() -> None:
    text = TABLE_PATH.read_text(encoding="utf-8")

    # 1. Replace the colspec (covers both the original and any
    #    previously-fixed single-line version).
    new_text, n_sub = COLSPEC_RE.subn(NEW_COLSPEC, text, count=1)
    if n_sub == 0:
        print(
            "WARNING: \\begin{longtable}{...} not found; "
            "colspec NOT changed. Please check the file manually."
        )
    else:
        print("Column specification replaced with fully top-aligned version.")
    text = new_text

    # 2. Insert \renewcommand{\arraystretch}{1.15} just before the table,
    #    if not already present.
    if ARRAYSTRETCH_LINE not in text:
        text = text.replace(
            "\\begin{longtable}",
            ARRAYSTRETCH_LINE + "\n\\begin{longtable}",
            1,
        )
        print("Added \\renewcommand{\\arraystretch}{1.15}.")
    else:
        print("\\arraystretch already set.")

    # 3. Insert \allowbreak after every \_ in columns 1-3 of data rows.
    lines = text.splitlines()
    out_lines = []
    n_changed = 0

    for line in lines:
        stripped = line.strip()
        is_structural = any(
            stripped.startswith(prefix) for prefix in SKIP_PREFIXES
        )
        if is_structural or not stripped:
            out_lines.append(line)
            continue

        new_line = add_breaks_to_first_three_columns(line)
        if new_line != line:
            n_changed += 1
        out_lines.append(new_line)

    TABLE_PATH.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"Processed {len(lines)} lines; {n_changed} data rows updated.")


if __name__ == "__main__":
    main()
