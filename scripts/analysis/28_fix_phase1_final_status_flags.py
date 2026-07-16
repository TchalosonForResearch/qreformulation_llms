#!/usr/bin/env python3
"""Repair inferential status fields in the final Phase 1 statistics tables."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ALPHA = 0.05
TARGET_DATASET = "bsard_test"
TARGET_FAMILY = "bsard_bm25_replacement"


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    root = args.root.resolve()
    stats_dir = root / "outputs" / "statistics"
    input_path = require(stats_dir / "canonical_statistics_final.csv")

    print("[1/4] Loading Phase 1 final table...")
    df = pd.read_csv(input_path)

    required = {
        "dataset",
        "experiment_family",
        "statistical_group",
        "method",
        "baseline_method",
        "metric",
        "mean_delta",
        "p_raw",
        "p_holm",
        "source_delta",
        "source_kind",
    }
    missing_columns = required.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df["mean_delta"] = pd.to_numeric(df["mean_delta"], errors="raise")
    df["p_raw"] = pd.to_numeric(df["p_raw"], errors="coerce")
    df["p_holm"] = pd.to_numeric(df["p_holm"], errors="coerce")
    df["source_delta"] = pd.to_numeric(df["source_delta"], errors="coerce")

    baseline = df["method"].astype(str).eq(df["baseline_method"].astype(str))
    has_p = df["p_holm"].notna()
    significant = has_p & df["p_holm"].lt(args.alpha)

    print("[2/4] Rebuilding status and nullable significance flags...")
    status = pd.Series("not_available", index=df.index, dtype="string")
    status.loc[baseline] = "baseline_not_tested"
    status.loc[~baseline & has_p & ~significant] = "non_significant"
    status.loc[~baseline & significant & df["mean_delta"].gt(0)] = (
        "significant_positive"
    )
    status.loc[~baseline & significant & df["mean_delta"].lt(0)] = (
        "significant_negative"
    )
    status.loc[~baseline & significant & df["mean_delta"].eq(0)] = (
        "significant_zero_direction"
    )
    df["inferential_status"] = status

    sig_any = pd.Series(pd.NA, index=df.index, dtype="boolean")
    sig_pos = pd.Series(pd.NA, index=df.index, dtype="boolean")
    sig_neg = pd.Series(pd.NA, index=df.index, dtype="boolean")

    tested = ~baseline & has_p
    sig_any.loc[tested] = significant.loc[tested]
    sig_pos.loc[tested] = (
        significant.loc[tested] & df.loc[tested, "mean_delta"].gt(0)
    )
    sig_neg.loc[tested] = (
        significant.loc[tested] & df.loc[tested, "mean_delta"].lt(0)
    )

    df["significant_holm_0.05"] = sig_any
    df["significant_positive_holm"] = sig_pos
    df["significant_negative_holm"] = sig_neg

    phase1c_source = df["source_kind"].astype(str).eq(
        "recomputed_missing_family"
    )
    fill_source_delta = phase1c_source & df["source_delta"].isna()
    df.loc[fill_source_delta, "source_delta"] = df.loc[
        fill_source_delta, "mean_delta"
    ]

    print("[3/4] Validating inferential coverage and BSARD results...")
    nonbaseline = ~baseline
    raw_covered = int((nonbaseline & df["p_raw"].notna()).sum())
    holm_covered = int((nonbaseline & df["p_holm"].notna()).sum())
    candidate_rows = int(nonbaseline.sum())

    if raw_covered != candidate_rows or holm_covered != candidate_rows:
        raise ValueError(
            "Incomplete final inferential coverage: "
            f"raw={raw_covered}/{candidate_rows}, "
            f"Holm={holm_covered}/{candidate_rows}"
        )

    target = df[
        df["dataset"].eq(TARGET_DATASET)
        & df["experiment_family"].eq(TARGET_FAMILY)
    ].copy()

    if len(target) != 24:
        raise ValueError(
            f"Expected 24 BSARD replacement rows, found {len(target)}"
        )

    target_sig = target[target["significant_holm_0.05"].fillna(False)].copy()
    expected = {
        ("bm25_deepseek_hyde_style", "Recall@100"),
        ("bm25_gpt_hyde_style", "Recall@100"),
    }
    observed = set(zip(target_sig["method"], target_sig["metric"]))
    if observed != expected:
        raise ValueError(
            f"Unexpected significant BSARD replacement set: {sorted(observed)}"
        )

    if not target_sig["significant_negative_holm"].fillna(False).all():
        raise ValueError(
            "The two significant BSARD replacement results must be negative."
        )

    # Rebuild coverage from the corrected final table.
    coverage = (
        df.assign(
            is_baseline=baseline,
            raw_available=df["p_raw"].notna(),
            holm_available=df["p_holm"].notna(),
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

    print("[4/4] Saving corrected final artifacts...")
    final_out = stats_dir / "canonical_statistics_final_corrected.csv"
    coverage_out = (
        stats_dir / "canonical_inferential_coverage_final_corrected.csv"
    )
    audit_out = stats_dir / "canonical_phase1_final_audit.txt"

    df.to_csv(final_out, index=False)
    coverage.to_csv(coverage_out, index=False)

    status_counts = (
        df["inferential_status"].value_counts(dropna=False).sort_index()
    )
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()

    with audit_out.open("w", encoding="utf-8") as handle:
        handle.write("FINAL PHASE 1 AUDIT\n")
        handle.write("===================\n\n")
        handle.write(
            "No retrieval metric, bootstrap interval, p-value, or Holm value "
            "was recomputed by this repair.\n"
        )
        handle.write(
            "Only inferential status/flag bookkeeping and the final coverage "
            "table were regenerated.\n\n"
        )
        handle.write(
            f"Input: {input_path.relative_to(root)}\n"
            f"Input SHA-256: {input_sha}\n"
            f"Candidate metric rows: {candidate_rows}\n"
            f"Raw p-value coverage: {raw_covered}/{candidate_rows}\n"
            f"Holm p-value coverage: {holm_covered}/{candidate_rows}\n"
            f"BSARD replacement rows: {len(target)}\n"
            f"Significant BSARD replacement rows: {len(target_sig)}\n\n"
        )
        handle.write("STATUS COUNTS\n")
        for label, count in status_counts.items():
            handle.write(f"- {label}: {int(count)}\n")

        handle.write("\nSIGNIFICANT BSARD REPLACEMENT RESULTS\n")
        for _, row in target_sig.sort_values(["method", "metric"]).iterrows():
            handle.write(
                f"- {row['method']} | {row['metric']} | "
                f"delta={row['mean_delta']:.6f} | "
                f"CI95=[{row['delta_ci95_low']:.6f}, "
                f"{row['delta_ci95_high']:.6f}] | "
                f"p_raw={row['p_raw']:.8g} | "
                f"p_holm={row['p_holm']:.8g} | "
                f"{row['inferential_status']}\n"
            )

    print("=" * 88)
    print("PHASE 1 FINAL BOOKKEEPING REPAIR COMPLETE")
    print("=" * 88)
    print(f"Candidate metric rows:     {candidate_rows}")
    print(f"Raw p-value coverage:      {raw_covered}/{candidate_rows}")
    print(f"Holm p-value coverage:     {holm_covered}/{candidate_rows}")
    print(f"Significant BSARD results: {len(target_sig)}")
    print("Saved:")
    print(f"  {final_out.relative_to(root)}")
    print(f"  {coverage_out.relative_to(root)}")
    print(f"  {audit_out.relative_to(root)}")


if __name__ == "__main__":
    main()
