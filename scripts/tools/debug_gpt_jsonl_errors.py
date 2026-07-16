"""Inspect malformed GPT reformulation records in JSONL outputs."""

from __future__ import annotations

import json
from pathlib import Path


GPT_RAW = Path(
    "data/processed/legalbench/rag_mini/reformulations/raw/gpt_mini.jsonl"
)

OUT_BAD = Path(
    "outputs/tables/legalbench/gpt_mini_invalid_json_lines.txt"
)

OUT_BAD.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not GPT_RAW.exists():
        raise FileNotFoundError(f"Missing file: {GPT_RAW}")

    bad_lines = []
    good_count = 0

    with GPT_RAW.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw = line.rstrip("\n")

            if not raw.strip():
                continue

            try:
                json.loads(raw)
                good_count += 1
            except json.JSONDecodeError as exc:
                bad_lines.append(
                    {
                        "line_number": line_number,
                        "error": str(exc),
                        "line": raw,
                    }
                )

    print("=" * 80)
    print("GPT JSONL validation")
    print("=" * 80)
    print(f"File: {GPT_RAW}")
    print(f"Valid JSON lines: {good_count}")
    print(f"Invalid JSON lines: {len(bad_lines)}")

    if bad_lines:
        print("\nInvalid lines:")
        for item in bad_lines[:20]:
            print("-" * 80)
            print(f"Line {item['line_number']}")
            print(f"Error: {item['error']}")
            print(item["line"][:1000])

        with OUT_BAD.open("w", encoding="utf-8") as out:
            for item in bad_lines:
                out.write("=" * 80 + "\n")
                out.write(f"Line {item['line_number']}\n")
                out.write(f"Error: {item['error']}\n")
                out.write(item["line"] + "\n")

        print(f"\nSaved invalid lines to: {OUT_BAD}")

    else:
        print("\nAll lines are valid JSON.")


if __name__ == "__main__":
    main()