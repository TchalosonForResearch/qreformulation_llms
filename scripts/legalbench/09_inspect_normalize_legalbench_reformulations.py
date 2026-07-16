"""Validate and normalize LegalBench reformulations from all generators."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed/legalbench/rag_mini")

INPUT_PATH = (
    DATA_DIR
    / "reformulations/input/legalbench_mini_queries_for_reformulation.jsonl"
)

RAW_DIR = DATA_DIR / "reformulations/raw"
NORMALIZED_DIR = DATA_DIR / "reformulations/normalized"
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

OUT_TABLE_DIR = Path("outputs/tables/legalbench")
OUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_DEEPSEEK_FILE = RAW_DIR / "deepseek_mini.jsonl"
DEFAULT_GPT_FILE = RAW_DIR / "gpt_mini.jsonl"

REQUIRED_TEXT_FIELDS = [
    "legal_rewrite",
    "keyword_expansion",
    "hyde_style",
]

ASSERTIVE_HYDE_PATTERNS = [
    "explicitly states",
    "expressly states",
    "expressly provides",
    "the agreement provides that",
    "the agreement states that",
    "the document states that",
    "the document provides that",
    "the clause provides that",
    "the clause states that",
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
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number} in {path}: {exc}"
                ) from exc

    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def detect_hyde_warning_terms(text: str) -> list[str]:
    lowered = (text or "").lower()
    return [pattern for pattern in ASSERTIVE_HYDE_PATTERNS if pattern in lowered]


def load_prepared_queries(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = read_jsonl(path)

    query_map = {}

    for row in rows:
        qid = str(row["query_id"])

        if qid in query_map:
            raise ValueError(f"Duplicate query_id in prepared input: {qid}")

        query_map[qid] = row

    return rows, query_map


def normalize_one_row(
    *,
    raw_row: dict,
    source_query: dict,
    generator: str,
    default_model: str,
    default_prompt_version: str,
    raw_file: Path,
) -> dict:
    query_id = str(source_query["query_id"])
    task = str(source_query.get("task", "unknown"))

    original_text = (
        source_query.get("original_text")
        or source_query.get("question")
        or source_query.get("text")
        or ""
    )

    missing_fields = []

    normalized_fields = {}

    for field in REQUIRED_TEXT_FIELDS:
        value = raw_row.get(field, "")

        if not isinstance(value, str) or not value.strip():
            missing_fields.append(field)
            normalized_fields[field] = ""
        else:
            normalized_fields[field] = value.strip()

    hyde_warning_terms = detect_hyde_warning_terms(
        normalized_fields.get("hyde_style", "")
    )

    source_status = str(raw_row.get("validation_status", "")).strip()

    if source_status == "api_error":
        validation_status = "api_error"
    elif missing_fields:
        validation_status = "invalid"
    else:
        validation_status = "valid"

    generator_model = (
        raw_row.get("generator_model")
        or raw_row.get("model")
        or default_model
    )

    prompt_version = (
        raw_row.get("prompt_version")
        or default_prompt_version
    )

    normalized = {
        "query_id": query_id,
        "task": task,
        "original_index": source_query.get("original_index"),
        "language": "en",
        "dataset": "legalbench_rag_mini",
        "original_text": original_text,
        "question": original_text,
        "generator": generator,
        "generator_model": generator_model,
        "prompt_version": prompt_version,
        "legal_rewrite": normalized_fields["legal_rewrite"],
        "keyword_expansion": normalized_fields["keyword_expansion"],
        "hyde_style": normalized_fields["hyde_style"],
        "validation_status": validation_status,
        "missing_fields": missing_fields,
        "hyde_style_warning_terms": hyde_warning_terms,
        "source_raw_file": str(raw_file),
    }

    if "attempts" in raw_row:
        normalized["attempts"] = raw_row.get("attempts")

    if "error" in raw_row:
        normalized["error"] = raw_row.get("error")

    return normalized


def normalize_generator_file(
    *,
    generator: str,
    raw_file: Path,
    query_map: dict[str, dict],
    default_model: str,
    default_prompt_version: str,
) -> tuple[list[dict], dict, list[dict]]:
    print("\n" + "=" * 80)
    print(f"Normalizing generator: {generator}")
    print("=" * 80)
    print(f"Raw file: {raw_file}")

    raw_rows = read_jsonl(raw_file)

    expected_qids = set(query_map.keys())
    raw_qids = []

    for row in raw_rows:
        qid = row.get("query_id")

        if qid is not None:
            raw_qids.append(str(qid))

    raw_counter = Counter(raw_qids)

    duplicate_qids = sorted(
        [qid for qid, count in raw_counter.items() if count > 1]
    )

    extra_qids = sorted(set(raw_qids) - expected_qids)

    normalized_rows = []
    warning_rows = []

    seen_output_qids = set()

    for row_index, raw_row in enumerate(raw_rows):
        qid = raw_row.get("query_id")

        if qid is None:
            warning_rows.append(
                {
                    "generator": generator,
                    "warning_type": "missing_query_id",
                    "query_id": "",
                    "raw_row_index": row_index,
                    "details": "Raw row has no query_id.",
                }
            )
            continue

        qid = str(qid)

        if qid not in query_map:
            warning_rows.append(
                {
                    "generator": generator,
                    "warning_type": "extra_query_id",
                    "query_id": qid,
                    "raw_row_index": row_index,
                    "details": "query_id not found in prepared input.",
                }
            )
            continue

        if qid in seen_output_qids:
            warning_rows.append(
                {
                    "generator": generator,
                    "warning_type": "duplicate_query_id",
                    "query_id": qid,
                    "raw_row_index": row_index,
                    "details": "Duplicate raw query_id skipped; first occurrence kept.",
                }
            )
            continue

        normalized = normalize_one_row(
            raw_row=raw_row,
            source_query=query_map[qid],
            generator=generator,
            default_model=default_model,
            default_prompt_version=default_prompt_version,
            raw_file=raw_file,
        )

        normalized_rows.append(normalized)
        seen_output_qids.add(qid)

        if normalized["validation_status"] != "valid":
            warning_rows.append(
                {
                    "generator": generator,
                    "warning_type": normalized["validation_status"],
                    "query_id": qid,
                    "raw_row_index": row_index,
                    "details": f"Missing fields: {normalized['missing_fields']}",
                }
            )

        if normalized["hyde_style_warning_terms"]:
            warning_rows.append(
                {
                    "generator": generator,
                    "warning_type": "hyde_style_assertive_terms",
                    "query_id": qid,
                    "raw_row_index": row_index,
                    "details": ", ".join(normalized["hyde_style_warning_terms"]),
                }
            )

    normalized_qids = set(row["query_id"] for row in normalized_rows)
    missing_qids = sorted(expected_qids - normalized_qids)

    num_valid = sum(
        1 for row in normalized_rows if row["validation_status"] == "valid"
    )

    num_invalid = sum(
        1 for row in normalized_rows if row["validation_status"] != "valid"
    )

    num_hyde_warnings = sum(
        1 for row in normalized_rows if row["hyde_style_warning_terms"]
    )

    summary = {
        "generator": generator,
        "raw_file": str(raw_file),
        "expected_queries": len(expected_qids),
        "raw_rows": len(raw_rows),
        "normalized_rows": len(normalized_rows),
        "valid_rows": num_valid,
        "invalid_rows": num_invalid,
        "missing_query_ids_count": len(missing_qids),
        "extra_query_ids_count": len(extra_qids),
        "duplicate_query_ids_count": len(duplicate_qids),
        "hyde_style_warning_rows": num_hyde_warnings,
        "missing_query_ids": missing_qids,
        "extra_query_ids": extra_qids,
        "duplicate_query_ids": duplicate_qids,
    }

    print("\nSummary:")
    printable_summary = {
        key: value
        for key, value in summary.items()
        if key not in {"missing_query_ids", "extra_query_ids", "duplicate_query_ids"}
    }
    print(json.dumps(printable_summary, indent=2, ensure_ascii=False))

    if missing_qids:
        print(f"First missing query_ids: {missing_qids[:10]}")

    if extra_qids:
        print(f"Extra query_ids: {extra_qids[:10]}")

    if duplicate_qids:
        print(f"Duplicate query_ids: {duplicate_qids[:10]}")

    return normalized_rows, summary, warning_rows


def make_quality_summary(normalized_rows: list[dict]) -> pd.DataFrame:
    rows = []

    for row in normalized_rows:
        rows.append(
            {
                "generator": row["generator"],
                "query_id": row["query_id"],
                "task": row["task"],
                "validation_status": row["validation_status"],
                "legal_rewrite_chars": len(row.get("legal_rewrite", "")),
                "keyword_expansion_chars": len(row.get("keyword_expansion", "")),
                "hyde_style_chars": len(row.get("hyde_style", "")),
                "hyde_style_warning_count": len(row.get("hyde_style_warning_terms", [])),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    summary = (
        df
        .groupby(["generator", "task"])
        .agg(
            num_rows=("query_id", "count"),
            valid_rows=("validation_status", lambda x: int((x == "valid").sum())),
            avg_legal_rewrite_chars=("legal_rewrite_chars", "mean"),
            avg_keyword_expansion_chars=("keyword_expansion_chars", "mean"),
            avg_hyde_style_chars=("hyde_style_chars", "mean"),
            hyde_style_warning_rows=("hyde_style_warning_count", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and normalize LegalBench-RAG mini reformulations."
    )

    parser.add_argument(
        "--deepseek-file",
        type=Path,
        default=DEFAULT_DEEPSEEK_FILE,
        help="Raw DeepSeek JSONL file.",
    )

    parser.add_argument(
        "--gpt-file",
        type=Path,
        default=DEFAULT_GPT_FILE,
        help="Raw GPT JSONL file.",
    )

    parser.add_argument(
        "--allow-missing-generators",
        action="store_true",
        help="Do not fail if one generator file is missing.",
    )

    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Do not fail if some query_ids are missing or rows are invalid.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing prepared input file: {INPUT_PATH}\n"
            "Run script 07_prepare_queries_for_reformulation.py first."
        )

    print("=" * 80)
    print("Inspecting and normalizing LegalBench-RAG mini reformulations")
    print("=" * 80)

    prepared_rows, query_map = load_prepared_queries(INPUT_PATH)

    print(f"Prepared queries: {len(prepared_rows)}")
    print(f"Input file: {INPUT_PATH}")

    generator_configs = [
        {
            "generator": "deepseek",
            "raw_file": args.deepseek_file,
            "default_model": "deepseek-v4-flash",
            "default_prompt_version": "legalbench_v2_deepseek",
            "normalized_file": NORMALIZED_DIR / "deepseek_mini.jsonl",
        },
        {
            "generator": "gpt",
            "raw_file": args.gpt_file,
            "default_model": "gpt_interface_manual",
            "default_prompt_version": "legalbench_v2_gpt",
            "normalized_file": NORMALIZED_DIR / "gpt_mini.jsonl",
        },
    ]

    all_normalized_rows = []
    all_summaries = []
    all_warnings = []
    missing_generator_files = []

    for config in generator_configs:
        raw_file = config["raw_file"]

        if not raw_file.exists():
            missing_generator_files.append(str(raw_file))

            message = f"Missing raw file for {config['generator']}: {raw_file}"

            if args.allow_missing_generators:
                print("\nWARNING:", message)
                continue

            raise FileNotFoundError(
                message
                + "\nUse --allow-missing-generators if you only want to inspect available files."
            )

        normalized_rows, summary, warning_rows = normalize_generator_file(
            generator=config["generator"],
            raw_file=raw_file,
            query_map=query_map,
            default_model=config["default_model"],
            default_prompt_version=config["default_prompt_version"],
        )

        write_jsonl(config["normalized_file"], normalized_rows)

        print(f"Saved normalized file to: {config['normalized_file']}")

        all_normalized_rows.extend(normalized_rows)
        all_summaries.append(summary)
        all_warnings.extend(warning_rows)

    all_generators_path = NORMALIZED_DIR / "all_generators_mini.jsonl"
    all_normalized_rows = sorted(
        all_normalized_rows,
        key=lambda row: (row["query_id"], row["generator"]),
    )
    write_jsonl(all_generators_path, all_normalized_rows)

    coverage_summary_path = (
        OUT_TABLE_DIR / "legalbench_reformulations_coverage_summary.csv"
    )
    quality_summary_path = (
        OUT_TABLE_DIR / "legalbench_reformulations_quality_summary.csv"
    )
    warnings_path = (
        OUT_TABLE_DIR / "legalbench_reformulations_warnings.csv"
    )
    missing_extra_path = (
        OUT_TABLE_DIR / "legalbench_reformulations_missing_extra_summary.json"
    )

    coverage_df = pd.DataFrame(
        [
            {
                key: value
                for key, value in summary.items()
                if key not in {
                    "missing_query_ids",
                    "extra_query_ids",
                    "duplicate_query_ids",
                }
            }
            for summary in all_summaries
        ]
    )

    coverage_df.to_csv(coverage_summary_path, index=False)

    quality_df = make_quality_summary(all_normalized_rows)
    quality_df.to_csv(quality_summary_path, index=False)

    warnings_df = pd.DataFrame(all_warnings)
    warnings_df.to_csv(warnings_path, index=False)

    missing_extra_payload = {
        "prepared_queries": len(prepared_rows),
        "missing_generator_files": missing_generator_files,
        "generators": {
            summary["generator"]: {
                "missing_query_ids": summary["missing_query_ids"],
                "extra_query_ids": summary["extra_query_ids"],
                "duplicate_query_ids": summary["duplicate_query_ids"],
            }
            for summary in all_summaries
        },
    }

    missing_extra_path.write_text(
        json.dumps(missing_extra_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print("Coverage summary")
    print("=" * 80)
    if coverage_df.empty:
        print("No coverage summary generated.")
    else:
        print(coverage_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Quality summary")
    print("=" * 80)
    if quality_df.empty:
        print("No quality summary generated.")
    else:
        print(quality_df.to_string(index=False))

    print("\nSaved files:")
    print(all_generators_path)
    print(coverage_summary_path)
    print(quality_summary_path)
    print(warnings_path)
    print(missing_extra_path)

    blocking_issues = []

    for summary in all_summaries:
        if summary["missing_query_ids_count"] > 0:
            blocking_issues.append(
                f"{summary['generator']}: missing query_ids = "
                f"{summary['missing_query_ids_count']}"
            )

        if summary["extra_query_ids_count"] > 0:
            blocking_issues.append(
                f"{summary['generator']}: extra query_ids = "
                f"{summary['extra_query_ids_count']}"
            )

        if summary["duplicate_query_ids_count"] > 0:
            blocking_issues.append(
                f"{summary['generator']}: duplicate query_ids = "
                f"{summary['duplicate_query_ids_count']}"
            )

        if summary["invalid_rows"] > 0:
            blocking_issues.append(
                f"{summary['generator']}: invalid rows = "
                f"{summary['invalid_rows']}"
            )

    if missing_generator_files and not args.allow_missing_generators:
        blocking_issues.append(
            "Missing generator files: " + ", ".join(missing_generator_files)
        )

    if blocking_issues and not args.allow_incomplete:
        print("\nBlocking issues detected:")
        for issue in blocking_issues:
            print(f"  - {issue}")

        raise SystemExit(
            "\nNormalization completed, but blocking issues were detected. "
            "Fix them or rerun with --allow-incomplete for inspection only."
        )

    print("\nNormalization completed successfully.")

    print("\nNext:")
    print("  scripts/legalbench/10_bm25_legalbench_reformulations.py")


if __name__ == "__main__":
    main()