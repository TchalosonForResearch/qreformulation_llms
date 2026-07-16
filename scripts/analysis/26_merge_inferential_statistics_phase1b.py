#!/usr/bin/env python3
"""Merge experiment-level inferential results into the canonical Phase 1 statistics table."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


METRICS = ("Recall@10", "Recall@100", "MRR@10", "nDCG@10")
ALPHA = 0.05
DELTA_TOL = 5e-6


def require_file(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {relative_path}")
    return path


def read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    return pd.read_csv(require_file(root, relative_path))


def as_number(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan
    return float(value)


def normalize_metric(value: object) -> str:
    text = str(value).strip()
    aliases = {
        "recall@10": "Recall@10",
        "recall_at_10": "Recall@10",
        "recall@100": "Recall@100",
        "recall_at_100": "Recall@100",
        "mrr@10": "MRR@10",
        "mrr_at_10": "MRR@10",
        "ndcg@10": "nDCG@10",
        "ndcg_at_10": "nDCG@10",
    }
    key = text.lower().replace(" ", "_")
    return aliases.get(key, text)


def normalize_multiview_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 1A accidentally placed both RRF and RCS multi-view methods under
    the label legalbench_multiview_rrf. Correct only the descriptive family
    label while preserving the combined 44-test Holm family.
    """
    out = df.copy()
    mask = out["method"].astype(str).str.startswith(("rrf_", "rcs_"))
    old_family = out["experiment_family"].astype(str).eq("legalbench_multiview_rrf")
    target = mask & old_family

    rcs = target & out["method"].astype(str).str.startswith("rcs_")
    rrf = target & out["method"].astype(str).str.startswith("rrf_")

    out.loc[rcs, "experiment_family"] = "legalbench_multiview_rcs"
    out.loc[rrf, "experiment_family"] = "legalbench_multiview_rrf"
    out.loc[target, "statistical_group"] = "legalbench_multiview_rrf_rcs"
    return out


def add_record(
    records: List[Dict[str, object]],
    *,
    dataset: str,
    experiment_family: str,
    statistical_group: str,
    holm_family_id: str,
    method: str,
    metric: str,
    p_raw: object,
    p_holm: object,
    source_path: str,
    source_delta: object = np.nan,
    source_kind: str = "experiment_stats",
    family_definition: str = "",
    note: str = "",
) -> None:
    records.append(
        {
            "dataset": dataset,
            "experiment_family": experiment_family,
            "statistical_group": statistical_group,
            "holm_family_id": holm_family_id,
            "method": str(method),
            "metric": normalize_metric(metric),
            "p_raw": as_number(p_raw),
            "p_holm": as_number(p_holm),
            "source_delta": as_number(source_delta),
            "inferential_source_path": source_path,
            "source_kind": source_kind,
            "family_definition": family_definition,
            "source_note": note,
        }
    )


def append_long_stats(
    records: List[Dict[str, object]],
    df: pd.DataFrame,
    *,
    dataset: str,
    experiment_family: str,
    statistical_group: str,
    holm_family_id: str,
    source_path: str,
    method_builder,
    p_raw_col: str = "p_raw_sign_flip",
    p_holm_col: str = "p_holm",
    delta_col: Optional[str] = "mean_gain",
    family_definition: str,
) -> None:
    for _, row in df.iterrows():
        add_record(
            records,
            dataset=dataset,
            experiment_family=experiment_family,
            statistical_group=statistical_group,
            holm_family_id=holm_family_id,
            method=method_builder(row),
            metric=row["metric"],
            p_raw=row[p_raw_col] if p_raw_col in row else np.nan,
            p_holm=row[p_holm_col] if p_holm_col in row else np.nan,
            source_delta=row[delta_col] if delta_col and delta_col in row else np.nan,
            source_path=source_path,
            family_definition=family_definition,
        )


def append_wide_summary(
    records: List[Dict[str, object]],
    df: pd.DataFrame,
    *,
    dataset: str,
    default_experiment_family: str,
    default_statistical_group: str,
    default_holm_family_id: str,
    source_path: str,
    family_definition: str,
    method_col: str = "method",
    group_col: Optional[str] = None,
    baseline_methods: Optional[Iterable[str]] = None,
) -> None:
    baselines = set(str(x) for x in (baseline_methods or []))
    for _, row in df.iterrows():
        method = str(row[method_col])
        if method in baselines:
            continue
        group = (
            str(row[group_col])
            if group_col and group_col in row and not pd.isna(row[group_col])
            else default_statistical_group
        )
        for metric in METRICS:
            add_record(
                records,
                dataset=dataset,
                experiment_family=default_experiment_family,
                statistical_group=group,
                holm_family_id=group if group_col else default_holm_family_id,
                method=method,
                metric=metric,
                p_raw=row.get(f"pRaw {metric}", np.nan),
                p_holm=row.get(f"pHolm {metric}", np.nan),
                source_delta=row.get(f"Delta {metric}", np.nan),
                source_path=source_path,
                family_definition=(
                    f"All candidate-versus-baseline metric tests in group {group}."
                    if group_col
                    else family_definition
                ),
            )


def append_bsard_synthesis_fallback(
    records: List[Dict[str, object]],
    df: pd.DataFrame,
    *,
    source_path: str,
) -> None:
    """
    BSARD direct replacement has no dedicated retained raw-p-value table.
    Only selected Holm values present in the final synthesis are imported.
    Recall@10 has no exact retained Holm value in this synthesis and remains NA.
    """
    label_to_method = {
        "DeepSeek keyword alone": "bm25_deepseek_keyword_expansion",
        "GPT keyword alone": "bm25_gpt_keyword_expansion",
    }
    for _, row in df.iterrows():
        label = str(row.get("method", ""))
        if label not in label_to_method:
            continue
        method = label_to_method[label]
        for metric in ("Recall@100", "MRR@10", "nDCG@10"):
            add_record(
                records,
                dataset="bsard_test",
                experiment_family="bsard_bm25_replacement",
                statistical_group="bsard_bm25_replacement",
                holm_family_id="bsard_final_synthesis_retained_holm",
                method=method,
                metric=metric,
                p_raw=np.nan,
                p_holm=row.get(f"pHolm {metric}", np.nan),
                source_delta=row.get(f"Delta {metric}", np.nan),
                source_path=source_path,
                source_kind="synthesis_fallback",
                family_definition=(
                    "Retained Holm values from the published BSARD synthesis; "
                    "the exact raw p-values and full original family membership "
                    "are not available in this artifact."
                ),
                note="Raw p-value unavailable; Recall@10 exact Holm value unavailable.",
            )


def deduplicate_and_validate_sources(
    source: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:
    keys = ["dataset", "method", "metric"]
    duplicates = source.duplicated(keys, keep=False)
    if not duplicates.any():
        return source

    resolved: List[pd.Series] = []
    for _, group in source.groupby(keys, sort=False, dropna=False):
        if len(group) == 1:
            resolved.append(group.iloc[0])
            continue

        raw_values = group["p_raw"].dropna().unique()
        holm_values = group["p_holm"].dropna().unique()
        if len(raw_values) > 1 or len(holm_values) > 1:
            warnings.append(
                "CONFLICTING INFERENTIAL SOURCES: "
                + " | ".join(str(x) for x in group[keys + [
                    "p_raw", "p_holm", "inferential_source_path"
                ]].to_dict("records"))
            )
            # Prefer a dedicated stats source over a synthesis fallback.
        ranked = group.assign(
            _priority=group["source_kind"].map(
                {"experiment_stats": 0, "wide_summary": 1, "synthesis_fallback": 2}
            ).fillna(9)
        ).sort_values("_priority")
        chosen = ranked.iloc[0].copy()

        # Fill a missing raw/Holm value from another non-conflicting source.
        if pd.isna(chosen["p_raw"]) and len(raw_values) == 1:
            chosen["p_raw"] = raw_values[0]
        if pd.isna(chosen["p_holm"]) and len(holm_values) == 1:
            chosen["p_holm"] = holm_values[0]
        resolved.append(chosen)

        warnings.append(
            f"Duplicate inferential mapping resolved for "
            f"{chosen['dataset']} | {chosen['method']} | {chosen['metric']}; "
            f"kept {chosen['inferential_source_path']}."
        )

    return pd.DataFrame(resolved).drop(columns=["_priority"], errors="ignore")


def infer_status(row: pd.Series) -> str:
    if pd.isna(row["p_holm"]):
        return "not_available"
    if float(row["p_holm"]) >= ALPHA:
        return "non_significant"
    delta = float(row["mean_delta"])
    if delta > 0:
        return "significant_positive"
    if delta < 0:
        return "significant_negative"
    return "significant_zero_direction"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--delta-tolerance",
        type=float,
        default=DELTA_TOL,
        help="Tolerance for canonical/source mean-delta consistency checks.",
    )
    parser.add_argument(
        "--standard-sign-flip-iterations",
        type=int,
        default=10_000,
        help=(
            "Iteration count used by the retained BSARD, LegalBench BM25, "
            "RRF/RCS, and chunk-only inferential runs. Change this only if "
            "those runs were executed with a non-default override."
        ),
    )
    parser.add_argument(
        "--dense-hybrid-sign-flip-iterations",
        type=int,
        default=100_000,
        help="Iteration count used by the final dense/hybrid inferential run.",
    )
    args = parser.parse_args()
    root = args.root.resolve()

    stats_dir = root / "outputs" / "statistics"
    stats_dir.mkdir(parents=True, exist_ok=True)

    canonical_long_path = "outputs/statistics/canonical_per_query_long.csv"
    canonical_summary_path = "outputs/statistics/canonical_descriptive_bootstrap.csv"
    canonical_manifest_path = "outputs/statistics/canonical_input_manifest.csv"

    print("[1/5] Loading and normalizing Phase 1A tables...")
    canonical_long = normalize_multiview_labels(read_csv(root, canonical_long_path))
    canonical = normalize_multiview_labels(read_csv(root, canonical_summary_path))
    input_manifest = read_csv(root, canonical_manifest_path)

    # Save the corrected family labels for all downstream analyses.
    normalized_long_out = stats_dir / "canonical_per_query_long_normalized.csv"
    canonical_long.to_csv(normalized_long_out, index=False)

    records: List[Dict[str, object]] = []
    warnings: List[str] = []

    print("[2/5] Loading retained inferential statistics...")

    # ------------------------------------------------------------------
    # BSARD one-view RRF: 6 methods x 4 metrics = 24 tests.
    # ------------------------------------------------------------------
    p = "outputs/tables/bsard/rrf_original_reformulation_stats.csv"
    df = read_csv(root, p)
    append_long_stats(
        records,
        df,
        dataset="bsard_test",
        experiment_family="bsard_rrf_one_view",
        statistical_group="bsard_rrf_one_view",
        holm_family_id="bsard_rrf_one_view",
        source_path=p,
        method_builder=lambda r: (
            f"rrf_original_{r['generator']}_{r['query_type']}"
        ),
        family_definition="Six one-view RRF candidates multiplied by four metrics (24 tests).",
    )

    # BSARD multi-view RRF: 3 methods x 4 metrics = 12 tests.
    p = "outputs/tables/bsard/rrf_all_reformulations_stats.csv"
    df = read_csv(root, p)
    append_long_stats(
        records,
        df,
        dataset="bsard_test",
        experiment_family="bsard_rrf_multi_view",
        statistical_group="bsard_rrf_multi_view",
        holm_family_id="bsard_rrf_multi_view",
        source_path=p,
        method_builder=lambda r: r["method"],
        family_definition="Three multi-view RRF candidates multiplied by four metrics (12 tests).",
    )

    # BSARD selected RCS: 5 methods x 4 metrics = 20 tests.
    p = "outputs/tables/bsard/rcs_selected_grid_stats.csv"
    df = read_csv(root, p)
    append_long_stats(
        records,
        df,
        dataset="bsard_test",
        experiment_family="bsard_rcs_selected",
        statistical_group="bsard_rcs_selected",
        holm_family_id="bsard_rcs_selected",
        source_path=p,
        method_builder=lambda r: r["method"],
        family_definition="Five selected RCS candidates multiplied by four metrics (20 tests).",
    )

    # BSARD direct replacement: only retained synthesis Holm values are available.
    p = "outputs/tables/bsard/bm25_final_synthesis_table_full.csv"
    append_bsard_synthesis_fallback(records, read_csv(root, p), source_path=p)

    # ------------------------------------------------------------------
    # LegalBench BM25 direct replacement: 6 methods x 4 = 24 tests.
    # ------------------------------------------------------------------
    p = "outputs/tables/legalbench/legalbench_replacement_summary_for_synthesis.csv"
    df = read_csv(root, p)
    append_long_stats(
        records,
        df,
        dataset="legalbench_mini",
        experiment_family="legalbench_bm25_replacement",
        statistical_group="legalbench_bm25_replacement",
        holm_family_id="legalbench_bm25_replacement",
        source_path=p,
        method_builder=lambda r: r["method"],
        delta_col="delta",
        family_definition="Six BM25 replacement candidates multiplied by four metrics (24 tests).",
    )

    # LegalBench one-view RRF: 6 methods x 4 = 24 tests.
    p = "outputs/tables/legalbench/rrf_original_reformulation_mini_stats.csv"
    df = read_csv(root, p)
    append_long_stats(
        records,
        df,
        dataset="legalbench_mini",
        experiment_family="legalbench_rrf_one_view",
        statistical_group="legalbench_rrf_one_view",
        holm_family_id="legalbench_rrf_one_view",
        source_path=p,
        method_builder=lambda r: r["method"],
        delta_col="mean_gain",
        family_definition="Six one-view RRF candidates multiplied by four metrics (24 tests).",
    )

    # LegalBench multi-view RRF and RCS share one combined 44-test Holm family.
    p = "outputs/tables/legalbench/multiview_rrf_rcs_mini_stats.csv"
    df = read_csv(root, p)
    for _, row in df.iterrows():
        method = str(row["method"])
        exp_family = (
            "legalbench_multiview_rcs"
            if method.startswith("rcs_")
            else "legalbench_multiview_rrf"
        )
        add_record(
            records,
            dataset="legalbench_mini",
            experiment_family=exp_family,
            statistical_group="legalbench_multiview_rrf_rcs",
            holm_family_id="legalbench_multiview_rrf_rcs",
            method=method,
            metric=row["metric"],
            p_raw=row.get("p_raw_sign_flip", np.nan),
            p_holm=row.get("p_holm", np.nan),
            source_delta=row.get("mean_gain", np.nan),
            source_path=p,
            family_definition=(
                "Eleven multi-view RRF/RCS candidates multiplied by four "
                "metrics in one combined Holm family (44 tests)."
            ),
        )

    # LegalBench chunk-only: baseline row is not tested.
    p = "outputs/tables/legalbench/chunk_only/legalbench_chunk_only_ablation_summary.csv"
    append_wide_summary(
        records,
        read_csv(root, p),
        dataset="legalbench_mini",
        default_experiment_family="legalbench_chunk_only",
        default_statistical_group="legalbench_chunk_only",
        default_holm_family_id="legalbench_chunk_only",
        source_path=p,
        family_definition="Five chunk-only candidates multiplied by four metrics (20 tests).",
        baseline_methods={"lb_chunk_original_bm25"},
    )
    for rec in records:
        if rec["inferential_source_path"] == p:
            rec["source_kind"] = "wide_summary"

    # Dense/hybrid: Holm correction is group-specific.
    p = (
        "outputs/tables/legalbench/dense_hybrid/"
        "legalbench_dense_hybrid_summary_filepath_chunk_models-e5-bge_dw0p5.csv"
    )
    dense_df = read_csv(root, p)
    baseline_methods = set(
        dense_df.loc[
            dense_df["method"].astype(str).str.endswith("_original"),
            "method",
        ].astype(str)
    )
    append_wide_summary(
        records,
        dense_df,
        dataset="legalbench_mini",
        default_experiment_family="legalbench_dense_hybrid",
        default_statistical_group="legalbench_dense_hybrid",
        default_holm_family_id="legalbench_dense_hybrid",
        source_path=p,
        family_definition="Group-specific dense/hybrid candidate-versus-baseline tests.",
        group_col="group",
        baseline_methods=baseline_methods,
    )
    for rec in records:
        if rec["inferential_source_path"] == p:
            rec["source_kind"] = "wide_summary"

    source = pd.DataFrame(records)
    source = deduplicate_and_validate_sources(source, warnings)

    # Record Monte Carlo sign-flip iteration metadata transparently.
    # The BSARD replacement fallback contains retained Holm values only;
    # its raw-test iteration count cannot be verified from that artifact.
    source["sign_flip_iterations"] = args.standard_sign_flip_iterations
    source["sign_flip_iterations_provenance"] = (
        "declared_standard_protocol_or_script_default"
    )

    dense_source_mask = source["inferential_source_path"].astype(str).str.contains(
        "dense_hybrid_summary", regex=False
    )
    source.loc[
        dense_source_mask, "sign_flip_iterations"
    ] = args.dense_hybrid_sign_flip_iterations
    source.loc[
        dense_source_mask, "sign_flip_iterations_provenance"
    ] = "verified_final_dense_hybrid_command"

    synthesis_fallback_mask = source["source_kind"].eq("synthesis_fallback")
    source.loc[synthesis_fallback_mask, "sign_flip_iterations"] = pd.NA
    source.loc[
        synthesis_fallback_mask, "sign_flip_iterations_provenance"
    ] = "not_verifiable_from_retained_synthesis_artifact"

    source["sign_flip_iterations"] = source[
        "sign_flip_iterations"
    ].astype("Int64")

    # Ensure unique mapping after conflict resolution.
    source_keys = ["dataset", "method", "metric"]
    if source.duplicated(source_keys).any():
        sample = source.loc[source.duplicated(source_keys, keep=False), source_keys]
        raise ValueError(
            "Inferential source still has duplicate mappings: "
            + str(sample.head(20).to_dict("records"))
        )

    print("[3/5] Merging inferential statistics with canonical summaries...")
    complete = canonical.merge(
        source,
        on=["dataset", "experiment_family", "statistical_group", "method", "metric"],
        how="left",
        validate="one_to_one",
    )

    # Mark baseline rows that correctly have no candidate-versus-baseline test.
    baseline_mask = complete["method"].eq(complete["baseline_method"])
    complete["inferential_status"] = complete.apply(infer_status, axis=1)
    complete.loc[baseline_mask, "inferential_status"] = "baseline_not_tested"
    complete["significant_holm_0.05"] = complete["p_holm"].lt(ALPHA)
    complete.loc[complete["p_holm"].isna(), "significant_holm_0.05"] = pd.NA
    complete["significant_positive_holm"] = (
        complete["p_holm"].lt(ALPHA) & complete["mean_delta"].gt(0)
    )
    complete["significant_negative_holm"] = (
        complete["p_holm"].lt(ALPHA) & complete["mean_delta"].lt(0)
    )

    print("[4/5] Auditing coverage, deltas, and Holm families...")
    nonbaseline = ~baseline_mask

    # Source-to-canonical delta checks.
    comparable = complete["source_delta"].notna()
    delta_error = (
        complete.loc[comparable, "source_delta"]
        - complete.loc[comparable, "mean_delta"]
    ).abs()
    bad_delta = delta_error > args.delta_tolerance
    if bad_delta.any():
        rows = complete.loc[comparable].loc[
            bad_delta,
            [
                "dataset", "experiment_family", "statistical_group",
                "method", "metric", "mean_delta", "source_delta",
                "inferential_source_path",
            ],
        ]
        warnings.append(
            f"{len(rows)} source/canonical mean-delta mismatches exceed "
            f"{args.delta_tolerance}. Examples: {rows.head(20).to_dict('records')}"
        )

    missing_raw = complete.loc[nonbaseline & complete["p_raw"].isna()]
    missing_holm = complete.loc[nonbaseline & complete["p_holm"].isna()]

    warnings.append(
        f"Non-baseline metric rows: {int(nonbaseline.sum())}."
    )
    warnings.append(
        f"Raw p-values available: "
        f"{int((nonbaseline & complete['p_raw'].notna()).sum())}; "
        f"missing: {len(missing_raw)}."
    )
    warnings.append(
        f"Holm p-values available: "
        f"{int((nonbaseline & complete['p_holm'].notna()).sum())}; "
        f"missing: {len(missing_holm)}."
    )

    if len(missing_holm):
        grouped_missing = (
            missing_holm.groupby(["dataset", "experiment_family"], dropna=False)
            .size()
            .reset_index(name="missing_holm_rows")
        )
        warnings.append(
            "Missing Holm coverage by family: "
            + grouped_missing.to_dict("records").__str__()
        )

    # Coverage table.
    coverage = (
        complete.assign(
            is_baseline=baseline_mask,
            raw_available=complete["p_raw"].notna(),
            holm_available=complete["p_holm"].notna(),
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

    # Holm family audit table.
    holm = source.groupby(
        [
            "dataset", "holm_family_id", "family_definition",
            "inferential_source_path", "source_kind",
        ],
        dropna=False,
        sort=True,
    ).agg(
        methods=("method", "nunique"),
        metrics=("metric", "nunique"),
        retained_rows=("metric", "size"),
        raw_p_rows=("p_raw", lambda s: int(s.notna().sum())),
        holm_p_rows=("p_holm", lambda s: int(s.notna().sum())),
        sign_flip_iterations=(
            "sign_flip_iterations",
            lambda s: "|".join(
                str(int(v)) for v in sorted(s.dropna().unique())
            ) if s.notna().any() else ""
        ),
        iteration_provenance=(
            "sign_flip_iterations_provenance",
            lambda s: "|".join(sorted(set(str(v) for v in s.dropna())))
        ),
    ).reset_index()

    # Add hashes of all inferential input files.
    used_paths = sorted(source["inferential_source_path"].dropna().unique())
    inferential_manifest_rows = []
    for rel in used_paths:
        path = require_file(root, rel)
        inferential_manifest_rows.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    inferential_manifest = pd.DataFrame(inferential_manifest_rows)

    print("[5/5] Saving Phase 1B outputs...")
    complete_out = stats_dir / "canonical_statistics_complete.csv"
    source_out = stats_dir / "canonical_inferential_sources.csv"
    coverage_out = stats_dir / "canonical_inferential_coverage.csv"
    holm_out = stats_dir / "canonical_holm_families.csv"
    infer_manifest_out = stats_dir / "canonical_inferential_input_manifest.csv"
    warnings_out = stats_dir / "canonical_phase1b_warnings.txt"

    complete.to_csv(complete_out, index=False)
    source.to_csv(source_out, index=False)
    coverage.to_csv(coverage_out, index=False)
    holm.to_csv(holm_out, index=False)
    inferential_manifest.to_csv(infer_manifest_out, index=False)

    with warnings_out.open("w", encoding="utf-8") as handle:
        handle.write(
            "PHASE 1B AUDIT\n"
            "================\n\n"
            "No p-value was recomputed by this script. Missing values remain NA.\n"
            f"Standard retained sign-flip runs are recorded as "
            f"{args.standard_sign_flip_iterations:,} iterations; the final "
            f"dense/hybrid run is recorded as "
            f"{args.dense_hybrid_sign_flip_iterations:,} iterations.\n"
            "BSARD replacement values imported only from the final synthesis "
            "retain an unknown iteration count because the raw inferential "
            "artifact is unavailable.\n"
            "The LegalBench multi-view RRF and RCS methods are separated as\n"
            "descriptive experiment families but retain one combined 44-test\n"
            "Holm family.\n\n"
        )
        for warning in warnings:
            handle.write(f"- {warning}\n")

        if len(missing_raw):
            handle.write("\nRAW P-VALUES NOT AVAILABLE\n")
            for _, row in missing_raw[
                ["dataset", "experiment_family", "method", "metric"]
            ].iterrows():
                handle.write(
                    f"- {row['dataset']} | {row['experiment_family']} | "
                    f"{row['method']} | {row['metric']}\n"
                )

        if len(missing_holm):
            handle.write("\nHOLM P-VALUES NOT AVAILABLE\n")
            for _, row in missing_holm[
                ["dataset", "experiment_family", "method", "metric"]
            ].iterrows():
                handle.write(
                    f"- {row['dataset']} | {row['experiment_family']} | "
                    f"{row['method']} | {row['metric']}\n"
                )

    print("=" * 88)
    print("PHASE 1B COMPLETE")
    print("=" * 88)
    print(f"Canonical rows with descriptive statistics: {len(complete):,}")
    print(f"Non-baseline metric rows:               {int(nonbaseline.sum()):,}")
    print(
        "Raw p-values available:                 "
        f"{int((nonbaseline & complete['p_raw'].notna()).sum()):,}"
    )
    print(
        "Holm p-values available:                "
        f"{int((nonbaseline & complete['p_holm'].notna()).sum()):,}"
    )
    print(f"Mean-delta mismatches:                  {int(bad_delta.sum()):,}")
    print(
        "Standard sign-flip iterations:          "
        f"{args.standard_sign_flip_iterations:,}"
    )
    print(
        "Dense/hybrid sign-flip iterations:      "
        f"{args.dense_hybrid_sign_flip_iterations:,}"
    )
    print("Saved:")
    for path in (
        normalized_long_out,
        complete_out,
        source_out,
        coverage_out,
        holm_out,
        infer_manifest_out,
        warnings_out,
    ):
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
