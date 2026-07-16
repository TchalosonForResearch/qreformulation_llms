"""Inspect the available BSARD query reformulation files and their schemas."""

from __future__ import annotations

import json
from pathlib import Path


RAW_QUERY_PATH = Path("data/raw/bsard/queries_test.jsonl")

REFORMULATION_FILES = {
    "deepseek": Path("data/processed/bsard/reformulations/raw/deepseek_test.jsonl"),
    "gpt": Path("data/processed/bsard/reformulations/raw/gpt_test.jsonl"),
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {e}"
                )

    return rows


def main() -> None:
    if not RAW_QUERY_PATH.exists():
        raise FileNotFoundError(f"Missing query file: {RAW_QUERY_PATH}")

    print("Loading BSARD test queries...")
    queries = read_jsonl(RAW_QUERY_PATH)
    expected_query_ids = {str(q["query_id"]) for q in queries}

    print(f"BSARD test queries: {len(expected_query_ids)}")

    for generator, path in REFORMULATION_FILES.items():
        print("\n" + "=" * 80)
        print(f"Inspecting generator: {generator}")
        print(f"File: {path}")

        if not path.exists():
            raise FileNotFoundError(
                f"Missing reformulation file for {generator}: {path}"
            )

        rows = read_jsonl(path)
        print(f"Rows: {len(rows)}")

        if not rows:
            print("File is empty.")
            continue

        print("\nFirst row keys:")
        print(list(rows[0].keys()))

        print("\nFirst row sample:")
        print(json.dumps(rows[0], ensure_ascii=False, indent=2)[:3000])

        # Try to detect query_id coverage
        query_ids = set()

        for row in rows:
            possible_qid = (
                row.get("query_id")
                or row.get("id")
                or row.get("qid")
                or row.get("question_id")
            )

            if possible_qid is not None:
                query_ids.add(str(possible_qid))

        print(f"\nDetected query IDs: {len(query_ids)}")

        missing = expected_query_ids - query_ids
        extra = query_ids - expected_query_ids

        print(f"Missing query IDs compared to BSARD test: {len(missing)}")
        print(f"Extra query IDs not in BSARD test: {len(extra)}")

        if missing:
            print("First missing IDs:", sorted(list(missing))[:20])

        if extra:
            print("First extra IDs:", sorted(list(extra))[:20])

        # Check expected fields
        expected_fields = ["legal_rewrite", "keyword_expansion", "hyde_style"]

        print("\nField presence check:")
        for field in expected_fields:
            count = sum(1 for row in rows if field in row and row.get(field))
            print(f"  {field}: {count}/{len(rows)}")


if __name__ == "__main__":
    main()