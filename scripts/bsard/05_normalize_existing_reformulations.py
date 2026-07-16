"""Normalize BSARD reformulations into a common query-level format."""

from __future__ import annotations

import json
from pathlib import Path


RAW_QUERY_PATH = Path("data/raw/bsard/queries_test.jsonl")

RAW_REFORMULATION_FILES = {
    "deepseek": Path("data/processed/bsard/reformulations/raw/deepseek_test.jsonl"),
    "gpt": Path("data/processed/bsard/reformulations/raw/gpt_test.jsonl"),
}

OUT_DIR = Path("data/processed/bsard/reformulations/normalized")
OUT_DIR.mkdir(parents=True, exist_ok=True)


REQUIRED_FIELDS = [
    "query_id",
    "legal_rewrite",
    "keyword_expansion",
    "hyde_style",
]


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


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_queries_by_id(path: Path) -> dict[str, dict]:
    queries = read_jsonl(path)
    return {str(q["query_id"]): q for q in queries}


def validate_required_fields(row: dict, path: Path, line_index: int) -> None:
    missing = []

    for field in REQUIRED_FIELDS:
        if field not in row or row[field] in [None, ""]:
            missing.append(field)

    if missing:
        raise ValueError(
            f"Missing required fields in {path}, row {line_index}: {missing}"
        )


def normalize_row(row: dict, generator_short: str, queries_by_id: dict[str, dict]) -> dict:
    qid = str(row["query_id"])

    if qid not in queries_by_id:
        raise ValueError(
            f"query_id {qid} found in reformulations but not in BSARD test queries."
        )

    query_obj = queries_by_id[qid]

    # Important :
    # original_text vient du fichier officiel BSARD queries_test.jsonl.
    # C'est le texte utilisé dans notre baseline BM25 original.
    original_text = query_obj.get("text") or query_obj.get("question") or ""

    return {
        "query_id": qid,
        "generator": generator_short,
        "generator_model": row.get("generator"),
        "prompt_version": row.get("prompt_version"),
        "validation_status": row.get("validation_status"),
        "attempts": row.get("attempts"),
        "original_text": original_text,
        "question": query_obj.get("question"),
        "extra_description": query_obj.get("extra_description"),
        "category": query_obj.get("category"),
        "subcategory": query_obj.get("subcategory"),
        "raw_original_from_reformulation_file": row.get("original"),
        "legal_rewrite": row.get("legal_rewrite", "").strip(),
        "keyword_expansion": row.get("keyword_expansion", "").strip(),
        "hyde_style": row.get("hyde_style", "").strip(),
    }


def main() -> None:
    if not RAW_QUERY_PATH.exists():
        raise FileNotFoundError(f"Missing query file: {RAW_QUERY_PATH}")

    print("Loading BSARD test queries...")
    queries_by_id = load_queries_by_id(RAW_QUERY_PATH)
    expected_query_ids = set(queries_by_id.keys())

    print(f"BSARD test queries: {len(expected_query_ids)}")

    all_normalized_rows = []

    for generator_short, input_path in RAW_REFORMULATION_FILES.items():
        print("\n" + "=" * 80)
        print(f"Normalizing generator: {generator_short}")
        print(f"Input file: {input_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Missing file: {input_path}")

        raw_rows = read_jsonl(input_path)
        print(f"Raw rows: {len(raw_rows)}")

        normalized_rows = []

        for idx, row in enumerate(raw_rows, start=1):
            validate_required_fields(row, input_path, idx)
            normalized_rows.append(
                normalize_row(
                    row=row,
                    generator_short=generator_short,
                    queries_by_id=queries_by_id,
                )
            )

        found_query_ids = {row["query_id"] for row in normalized_rows}

        missing = expected_query_ids - found_query_ids
        extra = found_query_ids - expected_query_ids

        if missing:
            raise ValueError(
                f"{generator_short}: missing query IDs: {sorted(list(missing))[:20]}"
            )

        if extra:
            raise ValueError(
                f"{generator_short}: extra query IDs: {sorted(list(extra))[:20]}"
            )

        output_path = OUT_DIR / f"{generator_short}_test.jsonl"
        write_jsonl(normalized_rows, output_path)

        print(f"Normalized rows: {len(normalized_rows)}")
        print(f"Saved to: {output_path}")

        all_normalized_rows.extend(normalized_rows)

    combined_path = OUT_DIR / "all_generators_test.jsonl"
    write_jsonl(all_normalized_rows, combined_path)

    print("\n" + "=" * 80)
    print(f"Combined rows: {len(all_normalized_rows)}")
    print(f"Saved combined file to: {combined_path}")

    print("\nExample normalized row:")
    print(json.dumps(all_normalized_rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()