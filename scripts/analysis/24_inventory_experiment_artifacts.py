#!/usr/bin/env python3
"""Inventory experiment outputs and build the manifests used by the statistical audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PRIMARY_METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
DEFAULT_SCAN_DIRS = ["outputs", "runs"]
DEFAULT_EXCLUDES = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "node_modules",
}


@dataclass
class ArtifactRecord:
    path: str
    relative_path: str
    extension: str
    size_bytes: int
    size_mb: float
    modified_time: float
    sha256: str
    delimiter: str
    artifact_type: str
    experiment_family: str
    fully_loaded: bool
    row_count: int | None
    column_count: int
    columns: str
    metric_columns: str
    delta_columns: str
    baseline_columns: str
    pvalue_columns: str
    query_count: int | None
    method_count: int | None
    methods: str
    task_count: int | None
    tasks: str
    duplicate_method_query_rows: int | None
    missing_primary_metric_cells: int | None
    has_all_primary_metrics: bool
    has_query_id: bool
    has_method: bool
    has_task: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit experiment CSV/TSV artifacts without recomputing retrieval."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Project root. Default: current directory.",
    )
    parser.add_argument(
        "--scan-dir",
        action="append",
        dest="scan_dirs",
        default=None,
        help=(
            "Directory relative to --root to scan. May be supplied multiple times. "
            "Default: outputs and runs."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/audit"),
        help="Output directory relative to --root.",
    )
    parser.add_argument(
        "--max-full-load-mb",
        type=float,
        default=100.0,
        help=(
            "Maximum file size loaded fully with pandas. Larger files are sampled "
            "and line-counted. Default: 100 MB."
        ),
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=5000,
        help="Rows sampled from files too large to load fully. Default: 5000.",
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Also scan the data directory. Usually unnecessary for Phase 0.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def infer_delimiter(path: Path) -> str:
    if path.suffix.lower() == ".tsv":
        return "\t"

    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            sample = f.read(8192)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        return dialect.delimiter
    except Exception:
        return ","


def count_data_rows(path: Path) -> int | None:
    """Count physical lines minus one header line. Returns None on failure."""
    try:
        with path.open("rb") as handle:
            line_count = sum(1 for _ in handle)
        return max(line_count - 1, 0)
    except OSError:
        return None


def unique_preview(series: pd.Series, limit: int = 30) -> tuple[int, str]:
    values = sorted({str(v) for v in series.dropna().tolist()})
    preview = values[:limit]
    suffix = " | ..." if len(values) > limit else ""
    return len(values), " | ".join(preview) + suffix


def infer_experiment_family(relative_path: str) -> str:
    value = relative_path.lower().replace("\\", "/")
    dataset = "unknown"
    if "bsard" in value:
        dataset = "bsard"
    elif "legalbench" in value:
        dataset = "legalbench"

    branch = "general"
    for candidate in [
        "dense_hybrid",
        "chunk_only",
        "anchor",
        "rcs",
        "rrf",
        "reformulation",
        "bm25",
    ]:
        if candidate in value:
            branch = candidate
            break
    return f"{dataset}:{branch}"


def classify_artifact(columns: Iterable[str], filename: str) -> str:
    cols = set(columns)
    metric_cols = set(PRIMARY_METRICS) & cols
    lower_name = filename.lower()

    if {"query_id", "doc_id", "rank"}.issubset(cols):
        return "retrieval_run"

    if "query_id" in cols and metric_cols:
        if any(str(c).startswith("Delta ") for c in cols):
            return "per_query_with_deltas"
        return "per_query_metrics"

    if "task" in cols and "method" in cols and metric_cols:
        return "by_task_metrics"

    if "method" in cols and metric_cols:
        return "summary_metrics"

    if "per_query" in lower_name or "per-query" in lower_name:
        return "possible_per_query_unknown_schema"

    if "summary" in lower_name:
        return "possible_summary_unknown_schema"

    return "other_table"


def stringify_columns(columns: Iterable[str]) -> str:
    return " | ".join(str(c) for c in columns)


def read_table(
    path: Path,
    delimiter: str,
    max_full_load_bytes: int,
    sample_rows: int,
) -> tuple[pd.DataFrame, bool, int | None, str]:
    size = path.stat().st_size
    fully_loaded = size <= max_full_load_bytes
    notes: list[str] = []

    read_kwargs: dict[str, Any] = {
        "sep": delimiter,
        "low_memory": False,
        "encoding": "utf-8-sig",
    }

    if fully_loaded:
        try:
            df = pd.read_csv(path, **read_kwargs)
            return df, True, int(len(df)), ""
        except UnicodeDecodeError:
            read_kwargs["encoding"] = "latin-1"
            df = pd.read_csv(path, **read_kwargs)
            notes.append("Loaded using latin-1 fallback.")
            return df, True, int(len(df)), " ".join(notes)
    else:
        read_kwargs["nrows"] = sample_rows
        try:
            df = pd.read_csv(path, **read_kwargs)
        except UnicodeDecodeError:
            read_kwargs["encoding"] = "latin-1"
            df = pd.read_csv(path, **read_kwargs)
            notes.append("Sample loaded using latin-1 fallback.")
        row_count = count_data_rows(path)
        notes.append(
            f"Sampled first {len(df)} rows because file exceeds full-load threshold."
        )
        return df, False, row_count, " ".join(notes)


def inspect_artifact(
    root: Path,
    path: Path,
    max_full_load_bytes: int,
    sample_rows: int,
) -> ArtifactRecord:
    delimiter = infer_delimiter(path)
    stat = path.stat()
    relative = path.relative_to(root).as_posix()

    try:
        df, fully_loaded, row_count, load_notes = read_table(
            path=path,
            delimiter=delimiter,
            max_full_load_bytes=max_full_load_bytes,
            sample_rows=sample_rows,
        )
    except Exception as exc:
        return ArtifactRecord(
            path=str(path.resolve()),
            relative_path=relative,
            extension=path.suffix.lower(),
            size_bytes=stat.st_size,
            size_mb=round(stat.st_size / (1024 * 1024), 4),
            modified_time=stat.st_mtime,
            sha256=sha256_file(path),
            delimiter="TAB" if delimiter == "\t" else delimiter,
            artifact_type="unreadable",
            experiment_family=infer_experiment_family(relative),
            fully_loaded=False,
            row_count=None,
            column_count=0,
            columns="",
            metric_columns="",
            delta_columns="",
            baseline_columns="",
            pvalue_columns="",
            query_count=None,
            method_count=None,
            methods="",
            task_count=None,
            tasks="",
            duplicate_method_query_rows=None,
            missing_primary_metric_cells=None,
            has_all_primary_metrics=False,
            has_query_id=False,
            has_method=False,
            has_task=False,
            notes=f"ERROR: {type(exc).__name__}: {exc}",
        )

    columns = [str(c) for c in df.columns]
    artifact_type = classify_artifact(columns, path.name)
    metric_columns = [c for c in columns if c in PRIMARY_METRICS]
    delta_columns = [c for c in columns if c.startswith("Delta ")]
    baseline_columns = [c for c in columns if c.startswith("baseline_")]
    pvalue_columns = [
        c
        for c in columns
        if c.startswith("pRaw ")
        or c.startswith("pHolm ")
        or c.lower() in {"p", "pvalue", "p_value", "p-value"}
    ]

    query_count: int | None = None
    method_count: int | None = None
    methods = ""
    task_count: int | None = None
    tasks = ""
    duplicate_count: int | None = None
    missing_primary_cells: int | None = None
    notes = [load_notes] if load_notes else []

    if "query_id" in df.columns:
        query_count = int(df["query_id"].astype(str).nunique(dropna=True))

    if "method" in df.columns:
        method_count, methods = unique_preview(df["method"])

    if "task" in df.columns:
        task_count, tasks = unique_preview(df["task"])

    if {"method", "query_id"}.issubset(df.columns):
        duplicate_count = int(df.duplicated(["method", "query_id"], keep=False).sum())
        if duplicate_count:
            notes.append(
                "Duplicate rows detected for the key (method, query_id); inspect before paired statistics."
            )

    present_primary = [m for m in PRIMARY_METRICS if m in df.columns]
    if present_primary:
        missing_primary_cells = int(df[present_primary].isna().sum().sum())
        if missing_primary_cells:
            notes.append(
                f"Missing cells in primary metrics: {missing_primary_cells}."
            )

    if artifact_type.startswith("per_query") and "method" not in df.columns:
        notes.append("Per-query-looking file has no method column.")

    if artifact_type.startswith("per_query") and not fully_loaded:
        notes.append(
            "Per-query file was sampled rather than fully loaded; rerun with a larger --max-full-load-mb if needed."
        )

    return ArtifactRecord(
        path=str(path.resolve()),
        relative_path=relative,
        extension=path.suffix.lower(),
        size_bytes=stat.st_size,
        size_mb=round(stat.st_size / (1024 * 1024), 4),
        modified_time=stat.st_mtime,
        sha256=sha256_file(path),
        delimiter="TAB" if delimiter == "\t" else delimiter,
        artifact_type=artifact_type,
        experiment_family=infer_experiment_family(relative),
        fully_loaded=fully_loaded,
        row_count=row_count,
        column_count=len(columns),
        columns=stringify_columns(columns),
        metric_columns=stringify_columns(metric_columns),
        delta_columns=stringify_columns(delta_columns),
        baseline_columns=stringify_columns(baseline_columns),
        pvalue_columns=stringify_columns(pvalue_columns),
        query_count=query_count,
        method_count=method_count,
        methods=methods,
        task_count=task_count,
        tasks=tasks,
        duplicate_method_query_rows=duplicate_count,
        missing_primary_metric_cells=missing_primary_cells,
        has_all_primary_metrics=all(m in df.columns for m in PRIMARY_METRICS),
        has_query_id="query_id" in df.columns,
        has_method="method" in df.columns,
        has_task="task" in df.columns,
        notes=" ".join(n for n in notes if n).strip(),
    )


def discover_files(root: Path, scan_dirs: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for relative_dir in scan_dirs:
        base = root / relative_dir
        if not base.exists():
            continue
        for suffix in ("*.csv", "*.tsv"):
            for path in base.rglob(suffix):
                if any(part in DEFAULT_EXCLUDES for part in path.parts):
                    continue
                paths.add(path)
    return sorted(paths, key=lambda p: p.as_posix().lower())


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No artifacts found._\n"

    # Keep the human-readable manifest compact. Full details remain in CSV/JSON.
    cols = [
        "relative_path",
        "artifact_type",
        "experiment_family",
        "row_count",
        "query_count",
        "method_count",
        "task_count",
        "has_all_primary_metrics",
        "notes",
    ]
    compact = df[cols].fillna("")
    return compact.to_markdown(index=False) + "\n"


def build_warnings(records: list[ArtifactRecord]) -> list[str]:
    warnings: list[str] = []

    if not records:
        return ["No CSV/TSV files were discovered."]

    per_query = [
        r
        for r in records
        if r.artifact_type in {"per_query_metrics", "per_query_with_deltas"}
    ]
    if not per_query:
        warnings.append(
            "No recognized per-query metric file was found. Phase 1 cannot be built until such files are located."
        )

    for r in records:
        if r.artifact_type == "unreadable":
            warnings.append(f"Unreadable file: {r.relative_path} — {r.notes}")
        if r.duplicate_method_query_rows:
            warnings.append(
                f"Duplicate (method, query_id) rows: {r.relative_path} — {r.duplicate_method_query_rows} rows involved."
            )
        if r.artifact_type.startswith("per_query") and not r.has_all_primary_metrics:
            warnings.append(
                f"Per-query candidate lacks one or more primary metrics: {r.relative_path}."
            )
        if r.artifact_type.startswith("per_query") and not r.has_method:
            warnings.append(
                f"Per-query candidate lacks a method column: {r.relative_path}."
            )

    families = {r.experiment_family for r in per_query}
    expected_family_markers = {
        "bsard": any(f.startswith("bsard:") for f in families),
        "legalbench": any(f.startswith("legalbench:") for f in families),
        "chunk_only": any("chunk_only" in f for f in families),
        "dense_hybrid": any("dense_hybrid" in f for f in families),
    }
    for marker, present in expected_family_markers.items():
        if not present:
            warnings.append(
                f"No recognized per-query artifact found for expected family marker: {marker}."
            )

    return warnings


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    scan_dirs = list(args.scan_dirs or DEFAULT_SCAN_DIRS)
    if args.include_data and "data" not in scan_dirs:
        scan_dirs.append("data")

    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(root, scan_dirs)
    max_full_load_bytes = int(args.max_full_load_mb * 1024 * 1024)

    records: list[ArtifactRecord] = []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] Inspecting {path.relative_to(root)}", flush=True)
        records.append(
            inspect_artifact(
                root=root,
                path=path,
                max_full_load_bytes=max_full_load_bytes,
                sample_rows=args.sample_rows,
            )
        )

    rows = [asdict(r) for r in records]
    manifest_df = pd.DataFrame(rows)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(
            ["artifact_type", "experiment_family", "relative_path"],
            kind="stable",
        ).reset_index(drop=True)

    manifest_csv = out_dir / "experiment_artifact_manifest.csv"
    manifest_md = out_dir / "experiment_artifact_manifest.md"
    inventory_json = out_dir / "experiment_artifact_inventory.json"
    per_query_csv = out_dir / "per_query_candidates.csv"
    warning_path = out_dir / "inventory_warnings.txt"

    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8")
    manifest_md.write_text(markdown_table(manifest_df), encoding="utf-8")

    inventory_payload = {
        "root": str(root),
        "scan_dirs": scan_dirs,
        "primary_metrics": PRIMARY_METRICS,
        "artifact_count": len(records),
        "artifacts": rows,
    }
    inventory_json.write_text(
        json.dumps(inventory_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if manifest_df.empty:
        per_query_df = manifest_df.copy()
    else:
        per_query_df = manifest_df[
            manifest_df["artifact_type"].isin(
                [
                    "per_query_metrics",
                    "per_query_with_deltas",
                    "possible_per_query_unknown_schema",
                ]
            )
        ].copy()
    per_query_df.to_csv(per_query_csv, index=False, encoding="utf-8")

    warnings = build_warnings(records)
    warning_path.write_text(
        "\n".join(f"- {item}" for item in warnings) + "\n",
        encoding="utf-8",
    )

    counts = (
        manifest_df["artifact_type"].value_counts().to_dict()
        if not manifest_df.empty
        else {}
    )
    print("\n" + "=" * 88)
    print("EXPERIMENT ARTIFACT INVENTORY")
    print("=" * 88)
    print(f"Project root: {root}")
    print(f"Files discovered: {len(records)}")
    for artifact_type, count in sorted(counts.items()):
        print(f"  {artifact_type}: {count}")
    print(f"Per-query candidates: {len(per_query_df)}")
    print(f"Warnings: {len(warnings)}")
    print("\nSaved:")
    for path in [manifest_csv, manifest_md, inventory_json, per_query_csv, warning_path]:
        print(f"  {path.relative_to(root)}")

    if warnings:
        print("\nImportant warnings:")
        for item in warnings[:20]:
            print(f"  - {item}")
        if len(warnings) > 20:
            print(f"  ... {len(warnings) - 20} additional warnings in {warning_path.relative_to(root)}")


if __name__ == "__main__":
    main()
