#!/usr/bin/env python3
"""Compute the missing paired inference for BSARD direct-replacement experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


DATASET = "bsard_test"
EXPERIMENT_FAMILY = "bsard_bm25_replacement"
STATISTICAL_GROUP = "bsard_bm25_replacement"
HOLM_FAMILY_ID = "bsard_bm25_replacement"
METRICS = ("Recall@10", "Recall@100", "MRR@10", "nDCG@10")
EPS = 1e-12
ALPHA = 0.05


def stable_int_hash(text: str, modulo: int | None = None) -> int:
    value = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return value % modulo if modulo else value


def sign_flip_pvalue(
    deltas: np.ndarray,
    *,
    iterations: int,
    seed: int,
    batch_size: int = 10_000,
) -> float:
    """Two-sided paired Monte Carlo sign-flip test, evaluated in batches."""
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 1 or len(deltas) == 0:
        raise ValueError("deltas must be a non-empty one-dimensional array")

    observed = float(np.mean(deltas))
    if np.all(np.abs(deltas) <= EPS):
        return 1.0

    rng = np.random.default_rng(seed)
    extreme = 0
    remaining = int(iterations)

    while remaining > 0:
        current = min(batch_size, remaining)
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(current, len(deltas)),
        )
        permuted_means = (signs * deltas[None, :]).mean(axis=1)
        extreme += int(np.sum(np.abs(permuted_means) >= abs(observed)))
        remaining -= current

    return float((extreme + 1.0) / (iterations + 1.0))


def holm_adjust(
    pvalues: Dict[Tuple[str, str], float]
) -> Dict[Tuple[str, str], float]:
    items = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(items)
    adjusted: Dict[Tuple[str, str], float] = {}
    running = 0.0

    for index, (key, pvalue) in enumerate(items):
        candidate = min((m - index) * float(pvalue), 1.0)
        running = max(running, candidate)
        adjusted[key] = min(running, 1.0)

    return adjusted


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def infer_status(delta: float, p_holm: float) -> str:
    if p_holm >= ALPHA:
        return "non_significant"
    if delta > 0:
        return "significant_positive"
    if delta < 0:
        return "significant_negative"
    return "significant_zero_direction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sign-flip-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args()

    if args.sign_flip_iterations <= 0:
        raise ValueError("--sign-flip-iterations must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    root = args.root.resolve()
    stats_dir = root / "outputs" / "statistics"

    normalized_long = stats_dir / "canonical_per_query_long_normalized.csv"
    original_long = stats_dir / "canonical_per_query_long.csv"
    long_path = normalized_long if normalized_long.exists() else original_long

    complete_path = require(stats_dir / "canonical_statistics_complete.csv")
    coverage_path = require(stats_dir / "canonical_inferential_coverage.csv")
    holm_path = require(stats_dir / "canonical_holm_families.csv")
    require(long_path)

    print("[1/5] Loading frozen canonical tables...")
    per_query = pd.read_csv(long_path)
    complete = pd.read_csv(complete_path)
    old_coverage = pd.read_csv(coverage_path)
    old_holm = pd.read_csv(holm_path)

    target = per_query[
        per_query["dataset"].eq(DATASET)
        & per_query["experiment_family"].eq(EXPERIMENT_FAMILY)
    ].copy()

    if target.empty:
        raise ValueError("No BSARD replacement rows found in the canonical table")

    expected_methods = sorted(target["method"].astype(str).unique())
    if len(expected_methods) != 6:
        raise ValueError(
            f"Expected 6 BSARD replacement methods, found {len(expected_methods)}: "
            f"{expected_methods}"
        )

    query_counts = target.groupby(["method", "metric"])["query_id"].nunique()
    if not (query_counts == 222).all():
        raise ValueError(
            "Every BSARD method/metric comparison must contain 222 unique queries. "
            f"Observed counts: {query_counts.to_dict()}"
        )

    duplicate_count = int(
        target.duplicated(["method", "metric", "query_id"]).sum()
    )
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate method/metric/query rows"
        )

    print("[2/5] Computing 24 paired sign-flip tests...")
    raw: Dict[Tuple[str, str], float] = {}
    means: Dict[Tuple[str, str], float] = {}

    for method in expected_methods:
        for metric_index, metric in enumerate(METRICS):
            rows = target[
                target["method"].eq(method) & target["metric"].eq(metric)
            ].sort_values("query_id")

            deltas = rows["delta"].to_numpy(dtype=float)
            key = (method, metric)
            means[key] = float(np.mean(deltas))
            raw[key] = sign_flip_pvalue(
                deltas,
                iterations=args.sign_flip_iterations,
                seed=(
                    args.seed
                    + 1000 * (metric_index + 1)
                    + stable_int_hash(method, modulo=997)
                ),
                batch_size=args.batch_size,
            )

    print("[3/5] Applying one Holm correction across 24 tests...")
    adjusted = holm_adjust(raw)

    inference_rows = []
    for method in expected_methods:
        for metric in METRICS:
            key = (method, metric)
            inference_rows.append(
                {
                    "dataset": DATASET,
                    "experiment_family": EXPERIMENT_FAMILY,
                    "statistical_group": STATISTICAL_GROUP,
                    "holm_family_id": HOLM_FAMILY_ID,
                    "method": method,
                    "metric": metric,
                    "mean_delta": means[key],
                    "p_raw": raw[key],
                    "p_holm": adjusted[key],
                    "sign_flip_iterations": args.sign_flip_iterations,
                    "sign_flip_seed": (
                        args.seed
                        + 1000 * (METRICS.index(metric) + 1)
                        + stable_int_hash(method, modulo=997)
                    ),
                    "sign_flip_iterations_provenance": (
                        "recomputed_from_frozen_canonical_per_query"
                    ),
                    "inferential_source_path": str(
                        long_path.relative_to(root)
                    ).replace("\\", "/"),
                    "source_kind": "recomputed_missing_family",
                    "family_definition": (
                        "Six BSARD direct-replacement candidates multiplied "
                        "by four metrics in one Holm family (24 tests)."
                    ),
                    "inferential_status": infer_status(
                        means[key], adjusted[key]
                    ),
                    "significant_holm_0.05": adjusted[key] < ALPHA,
                    "significant_positive_holm": (
                        adjusted[key] < ALPHA and means[key] > 0
                    ),
                    "significant_negative_holm": (
                        adjusted[key] < ALPHA and means[key] < 0
                    ),
                }
            )

    inference = pd.DataFrame(inference_rows)

    print("[4/5] Filling only missing canonical inferential cells...")
    keys = [
        "dataset",
        "experiment_family",
        "statistical_group",
        "method",
        "metric",
    ]
    target_complete = complete[
        complete["dataset"].eq(DATASET)
        & complete["experiment_family"].eq(EXPERIMENT_FAMILY)
    ].copy()

    if len(target_complete) != 24:
        raise ValueError(
            f"Expected 24 BSARD replacement summary rows, found "
            f"{len(target_complete)}"
        )

    if target_complete["p_raw"].notna().any() or target_complete["p_holm"].notna().any():
        raise ValueError(
            "Existing BSARD replacement p-values were found. "
            "This script refuses to overwrite them."
        )

    merged = complete.merge(
        inference,
        on=keys,
        how="left",
        suffixes=("", "_phase1c"),
        validate="one_to_one",
    )

    fill_columns = [
        "holm_family_id",
        "p_raw",
        "p_holm",
        "sign_flip_iterations",
        "sign_flip_iterations_provenance",
        "inferential_source_path",
        "source_kind",
        "family_definition",
        "inferential_status",
        "significant_holm_0.05",
        "significant_positive_holm",
        "significant_negative_holm",
    ]

    for column in fill_columns:
        phase_column = f"{column}_phase1c"
        if phase_column not in merged.columns:
            continue
        fill_mask = merged[column].isna() & merged[phase_column].notna()
        merged.loc[fill_mask, column] = merged.loc[fill_mask, phase_column]
        merged = merged.drop(columns=[phase_column])

    # Drop helper columns duplicated from the Phase 1C table.
    merged = merged.drop(
        columns=[
            "mean_delta_phase1c",
            "sign_flip_seed",
        ],
        errors="ignore",
    )

    # Validate complete inferential coverage for every non-baseline row.
    nonbaseline = ~merged["method"].eq(merged["baseline_method"])
    missing_raw = merged.loc[nonbaseline, "p_raw"].isna().sum()
    missing_holm = merged.loc[nonbaseline, "p_holm"].isna().sum()

    if missing_raw or missing_holm:
        raise ValueError(
            f"Inferential coverage remains incomplete: "
            f"missing raw={missing_raw}, missing Holm={missing_holm}"
        )

    # Rebuild coverage.
    coverage = (
        merged.assign(
            is_baseline=merged["method"].eq(merged["baseline_method"]),
            raw_available=merged["p_raw"].notna(),
            holm_available=merged["p_holm"].notna(),
        )
        .groupby(
            ["dataset", "experiment_family", "statistical_group"],
            dropna=False,
            sort=True,
        )
        .agg(
            methods=("method", "nunique"),
            metric_rows=("metric", "size"),
            baseline_rows=("is_baseline", "sum"),
            raw_p_rows=("raw_available", "sum"),
            holm_p_rows=("holm_available", "sum"),
        )
        .reset_index()
    )
    coverage["candidate_metric_rows"] = (
        coverage["metric_rows"] - coverage["baseline_rows"]
    )
    coverage["raw_coverage_rate"] = np.where(
        coverage["candidate_metric_rows"] > 0,
        coverage["raw_p_rows"] / coverage["candidate_metric_rows"],
        np.nan,
    )
    coverage["holm_coverage_rate"] = np.where(
        coverage["candidate_metric_rows"] > 0,
        coverage["holm_p_rows"] / coverage["candidate_metric_rows"],
        np.nan,
    )

    # Replace the empty synthesis fallback family with the actual recomputed family.
    holm = old_holm[
        ~old_holm["holm_family_id"].eq(
            "bsard_final_synthesis_retained_holm"
        )
    ].copy()

    new_holm = pd.DataFrame(
        [
            {
                "dataset": DATASET,
                "holm_family_id": HOLM_FAMILY_ID,
                "family_definition": (
                    "Six BSARD direct-replacement candidates multiplied "
                    "by four metrics in one Holm family (24 tests)."
                ),
                "inferential_source_path": str(
                    long_path.relative_to(root)
                ).replace("\\", "/"),
                "source_kind": "recomputed_missing_family",
                "methods": 6,
                "metrics": 4,
                "retained_rows": 24,
                "raw_p_rows": 24,
                "holm_p_rows": 24,
                "sign_flip_iterations": args.sign_flip_iterations,
                "iteration_provenance": (
                    "recomputed_from_frozen_canonical_per_query"
                ),
            }
        ]
    )
    holm = pd.concat([holm, new_holm], ignore_index=True).sort_values(
        ["dataset", "holm_family_id"]
    )

    print("[5/5] Saving final Phase 1 outputs...")
    inference_out = stats_dir / "canonical_bsard_replacement_inference.csv"
    final_out = stats_dir / "canonical_statistics_final.csv"
    coverage_out = stats_dir / "canonical_inferential_coverage_final.csv"
    holm_out = stats_dir / "canonical_holm_families_final.csv"
    audit_out = stats_dir / "canonical_phase1c_audit.txt"

    inference.to_csv(inference_out, index=False)
    merged.to_csv(final_out, index=False)
    coverage.to_csv(coverage_out, index=False)
    holm.to_csv(holm_out, index=False)

    significant = inference[inference["significant_holm_0.05"]].copy()

    with audit_out.open("w", encoding="utf-8") as handle:
        handle.write("PHASE 1C AUDIT\n")
        handle.write("================\n\n")
        handle.write(
            "The previously missing BSARD direct-replacement family was "
            "computed from the frozen canonical per-query deltas.\n"
        )
        handle.write(
            f"Sign-flip iterations: {args.sign_flip_iterations:,}\n"
        )
        handle.write(f"Base seed: {args.seed}\n")
        handle.write("Holm family size: 24 tests\n")
        handle.write(
            f"Final non-baseline rows with raw p-values: "
            f"{int((nonbaseline & merged['p_raw'].notna()).sum())}/"
            f"{int(nonbaseline.sum())}\n"
        )
        handle.write(
            f"Final non-baseline rows with Holm p-values: "
            f"{int((nonbaseline & merged['p_holm'].notna()).sum())}/"
            f"{int(nonbaseline.sum())}\n"
        )
        handle.write("\nSIGNIFICANT BSARD REPLACEMENT RESULTS\n")
        if significant.empty:
            handle.write("- None after Holm correction.\n")
        else:
            for _, row in significant.sort_values(
                ["method", "metric"]
            ).iterrows():
                handle.write(
                    f"- {row['method']} | {row['metric']} | "
                    f"delta={row['mean_delta']:.6f} | "
                    f"p_raw={row['p_raw']:.6g} | "
                    f"p_holm={row['p_holm']:.6g} | "
                    f"{row['inferential_status']}\n"
                )

    print("=" * 88)
    print("PHASE 1C COMPLETE")
    print("=" * 88)
    print(f"BSARD replacement tests:        {len(inference):,}")
    print(f"Final non-baseline metric rows: {int(nonbaseline.sum()):,}")
    print(
        "Raw p-value coverage:          "
        f"{int((nonbaseline & merged['p_raw'].notna()).sum()):,}/"
        f"{int(nonbaseline.sum()):,}"
    )
    print(
        "Holm p-value coverage:         "
        f"{int((nonbaseline & merged['p_holm'].notna()).sum()):,}/"
        f"{int(nonbaseline.sum()):,}"
    )
    print(f"Significant BSARD rows:         {len(significant):,}")
    print("Saved:")
    for path in (
        inference_out,
        final_out,
        coverage_out,
        holm_out,
        audit_out,
    ):
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
