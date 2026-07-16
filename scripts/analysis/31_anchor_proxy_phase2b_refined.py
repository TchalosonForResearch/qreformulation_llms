#!/usr/bin/env python3
"""Refine the LegalBench anchor proxy and run clustered association analyses."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-12
PRIMARY_SIGNAL = "combined_anchor_retention_v3"
METRIC_DELTAS = (
    "delta_Recall@10",
    "delta_Recall@100",
    "delta_MRR@10",
    "delta_nDCG@10",
)
TASK_ORDER = ("contractnli", "cuad", "maud", "privacy_qa")
QUERY_TYPES = ("legal_rewrite", "keyword_expansion", "hyde_style")
GENERATORS = ("deepseek", "gpt")

# Conservative exclusions learned from the Phase 2A validation audit.
# These are generic function words, legal-document descriptors, corporate
# suffixes, and product-title filler terms rather than source identifiers.
DOCUMENT_GENERIC = {
    "consider", "agreement", "agreements", "contract", "contracts",
    "document", "documents", "privacy", "policy", "policies",
    "non", "disclosure", "nondisclosure", "nda", "mutual",
    "acquisition", "merger", "branding", "cobranding", "co",
    "endorsement", "licensing", "license", "hosting", "content",
    "intellectual", "property", "development", "collaboration",
    "distributor", "distributorship", "reseller", "sponsorship",
    "event", "business", "promotion", "maintenance", "network",
    "build", "media", "parent", "target", "party", "parties",
    "other", "the", "a", "an", "and", "or", "of", "for", "to",
    "with", "between", "under", "on", "in", "at", "from",
    "university", "group", "services", "service", "financial",
    "insurance", "networks", "messenger", "app", "application",
    "stored", "value", "cards", "card", "day", "planner",
    "reminder", "list", "do",
}
CORPORATE_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "ltd",
    "limited", "plc", "bv", "bvi", "ag", "sa", "nv", "lp", "llp",
    "company", "co",
}
PATH_NOISE = {
    "txt", "pdf", "doc", "docx", "ex", "rev", "ka",
    "contractnli", "cuad", "maud", "privacyqa", "privacy_qa",
    "form", "final", "finalv3", "production", "uploads", "upload",
    "image", "direct",
}


def require_first(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the expected files exists:\n"
        + "\n".join(f"  - {path}" for path in paths)
    )


def ascii_lower(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def simple_tokens(value: object) -> List[str]:
    return [
        ascii_lower(token)
        for token in re.findall(r"[A-Za-z0-9]+", str(value))
    ]


def path_tokens(value: object) -> List[str]:
    """
    Extract filename tokens while retaining both raw compact forms and
    camel/alphanumeric components. Dates and numeric filing identifiers are
    excluded because they cannot count as retained original-query anchors
    unless they are explicitly mentioned in the query.
    """
    base = os.path.splitext(os.path.basename(str(value)))[0]
    raw = re.findall(r"[A-Za-z0-9]+", base)
    output: List[str] = []

    for token in raw:
        output.append(ascii_lower(token))
        split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", token)
        split = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", split)
        split = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", split)
        output.extend(simple_tokens(split))

    cleaned = {
        token
        for token in output
        if len(token) >= 2
        and not token.isdigit()
        and token not in DOCUMENT_GENERIC
        and token not in CORPORATE_SUFFIXES
        and token not in PATH_NOISE
    }
    return sorted(cleaned)


def entity_phrases(original_text: object) -> List[str]:
    """
    Parse source names conservatively from the regular query preamble.

    For "between X and Y for Z", only X and Y are treated as party names.
    The purpose/product phrase after "for" is excluded to avoid counting
    generic contract descriptors as source anchors.
    """
    prefix = str(original_text).split(";", 1)[0]
    phrases: List[str] = []

    # Quoted party or application names.
    phrases.extend(re.findall(r'["“](.+?)["”]', prefix))

    lower = prefix.lower()
    if " between " in lower:
        start = lower.index(" between ") + len(" between ")
        tail = prefix[start:]
        tail = re.split(r"\s+for\s+", tail, maxsplit=1, flags=re.I)[0]
        tail = re.sub(r"\b(?:Parent|Target)\b", " ", tail, flags=re.I)
        tail = tail.replace('"', "").replace("“", "").replace("”", "")
        parts = re.split(r"\s+and\s+", tail, flags=re.I)
        phrases.extend(part.strip(" ,") for part in parts if part.strip(" ,"))
    else:
        # Handles both "Fiverr's" and "M5-Systems'".
        match = re.search(
            r"^\s*Consider\s+(.+?)[\'’](?:s)?\s+",
            prefix,
            flags=re.I,
        )
        if match:
            phrases.append(match.group(1).strip(' "\''))

    deduplicated: List[str] = []
    seen = set()
    for phrase in phrases:
        key = ascii_lower(phrase).strip()
        if key and key not in seen:
            seen.add(key)
            deduplicated.append(phrase)
    return deduplicated


def entity_anchor_tokens(original_text: object) -> List[str]:
    anchors: List[str] = []

    for phrase in entity_phrases(original_text):
        # Preserve AT&T as ATT rather than the function word "at".
        normalized_phrase = re.sub(
            r"(?<=[A-Za-z])&(?=[A-Za-z])",
            "",
            phrase,
        )
        raw_tokens = re.findall(r"[A-Za-z0-9]+", normalized_phrase)
        underscore_or_hyphen_name = "_" in phrase or "-" in phrase

        for raw_token in raw_tokens:
            token = ascii_lower(raw_token)
            if len(token) < 2 or token in DOCUMENT_GENERIC:
                continue
            if token in CORPORATE_SUFFIXES:
                # Keep short all-caps components inside compound identifiers
                # such as SE_NDCA, but remove ordinary corporate suffixes.
                if not (underscore_or_hyphen_name and raw_token.isupper()):
                    continue
            anchors.append(token)

    return sorted(set(anchors))


def path_overlap_anchor_tokens(
    original_text: object,
    file_path: object,
) -> List[str]:
    """
    A filepath token is eligible only when it is an exact normalized token
    in the original query. Phase 2A allowed arbitrary compact substrings,
    which produced false anchors such as "form" from "information" and
    "rand" from "branding".
    """
    original_set = set(simple_tokens(original_text))
    return sorted(
        {
            token
            for token in path_tokens(file_path)
            if token in original_set
        }
    )


def token_is_retained(token: str, reformulation_text: object) -> bool:
    reform_set = set(simple_tokens(reformulation_text))
    reform_compact = re.sub(
        r"[^a-z0-9]+",
        "",
        ascii_lower(reformulation_text),
    )
    return token in reform_set or (len(token) >= 4 and token in reform_compact)


def retained_tokens(
    anchors: Sequence[str],
    reformulation_text: object,
) -> Tuple[List[str], List[str], float]:
    kept = [
        token for token in anchors
        if token_is_retained(token, reformulation_text)
    ]
    dropped = [token for token in anchors if token not in set(kept)]
    retention = len(kept) / len(anchors) if anchors else np.nan
    return kept, dropped, retention


def rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(dtype=float)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.std(x) <= EPS or np.std(y) <= EPS:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan
    return pearson(rank_average(x), rank_average(y))


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 4:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    n = len(x)

    for index in range(iterations):
        sample = rng.integers(0, n, size=n)
        estimates[index] = spearman(x[sample], y[sample])

    estimates = estimates[np.isfinite(estimates)]
    if not len(estimates):
        return np.nan, np.nan
    return (
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
    )


def approximate_spearman_pvalue(rho: float, n: int) -> float:
    """
    Two-sided large-sample approximation using the t transformation.
    This is used only in the subgroup sensitivity table. The primary pooled
    inference uses the cluster permutation test below.
    """
    if not np.isfinite(rho) or n < 4 or abs(rho) >= 1:
        if abs(rho) == 1 and n >= 4:
            return 0.0
        return np.nan

    try:
        from scipy.stats import t as student_t
    except Exception:
        return np.nan

    statistic = rho * math.sqrt((n - 2) / max(1 - rho * rho, EPS))
    return float(2 * student_t.sf(abs(statistic), df=n - 2))


def permutation_spearman_pvalue(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    observed = spearman(x, y)
    if not np.isfinite(observed):
        return np.nan

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        permuted = spearman(x, rng.permutation(y))
        if np.isfinite(permuted) and abs(permuted) >= abs(observed):
            extreme += 1
    return float((extreme + 1) / (iterations + 1))


def holm_adjust(values: Sequence[float]) -> List[float]:
    pvalues = np.asarray(values, dtype=float)
    adjusted = np.full(len(pvalues), np.nan, dtype=float)
    valid = np.where(np.isfinite(pvalues))[0]
    if not len(valid):
        return adjusted.tolist()

    ordered = valid[np.argsort(pvalues[valid])]
    running = 0.0
    m = len(ordered)
    for rank, position in enumerate(ordered):
        candidate = min((m - rank) * pvalues[position], 1.0)
        running = max(running, candidate)
        adjusted[position] = min(running, 1.0)
    return adjusted.tolist()


def within_stratum_rank_residuals(
    frame: pd.DataFrame,
    signal: str,
    outcome: str,
) -> pd.DataFrame:
    output = frame.copy()
    strata = ["generator", "query_type", "task"]

    output["_x_rank"] = output.groupby(strata, sort=False)[signal].transform(
        lambda values: values.rank(method="average")
    )
    output["_y_rank"] = output.groupby(strata, sort=False)[outcome].transform(
        lambda values: values.rank(method="average")
    )
    output["_x_resid"] = output["_x_rank"] - output.groupby(
        strata, sort=False
    )["_x_rank"].transform("mean")
    output["_y_resid"] = output["_y_rank"] - output.groupby(
        strata, sort=False
    )["_y_rank"].transform("mean")
    return output


def clustered_pooled_inference(
    frame: pd.DataFrame,
    signal: str,
    outcome: str,
    *,
    bootstrap_iterations: int,
    permutation_iterations: int,
    seed: int,
) -> Dict[str, float]:
    """
    Pooled within-stratum Spearman association.

    First, x and y are ranked and centred within each
    generator × query_type × task stratum. Then query clusters are resampled
    within task. A permutation applies the same query permutation to all six
    reformulation rows in a task, preserving within-query dependence.
    """
    clean = frame[
        frame[signal].notna() & frame[outcome].notna()
    ].copy()
    residual = within_stratum_rank_residuals(clean, signal, outcome)

    method_key = (
        residual["generator"].astype(str)
        + "::"
        + residual["query_type"].astype(str)
    )
    residual["_method_key"] = method_key
    method_order = sorted(residual["_method_key"].unique())

    matrices = {}
    total_clusters = 0
    for task in TASK_ORDER:
        task_rows = residual[residual["task"].eq(task)]
        x_matrix = task_rows.pivot(
            index="query_id", columns="_method_key", values="_x_resid"
        ).reindex(columns=method_order)
        y_matrix = task_rows.pivot(
            index="query_id", columns="_method_key", values="_y_resid"
        ).reindex(columns=method_order)

        common = x_matrix.index.intersection(y_matrix.index)
        x_matrix = x_matrix.loc[common]
        y_matrix = y_matrix.loc[common]

        if x_matrix.isna().any().any() or y_matrix.isna().any().any():
            raise ValueError(
                f"Incomplete query × method matrix for task {task}"
            )

        matrices[task] = (
            x_matrix.to_numpy(dtype=float),
            y_matrix.to_numpy(dtype=float),
        )
        total_clusters += len(common)

    x_all = np.concatenate([matrices[t][0].ravel() for t in TASK_ORDER])
    y_all = np.concatenate([matrices[t][1].ravel() for t in TASK_ORDER])
    observed = pearson(x_all, y_all)

    rng = np.random.default_rng(seed)

    bootstrap = np.empty(bootstrap_iterations, dtype=float)
    for index in range(bootstrap_iterations):
        x_parts = []
        y_parts = []
        for task in TASK_ORDER:
            x_matrix, y_matrix = matrices[task]
            sampled = rng.integers(0, x_matrix.shape[0], x_matrix.shape[0])
            x_parts.append(x_matrix[sampled].ravel())
            y_parts.append(y_matrix[sampled].ravel())
        bootstrap[index] = pearson(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
        )

    permutation = np.empty(permutation_iterations, dtype=float)
    for index in range(permutation_iterations):
        x_parts = []
        y_parts = []
        for task in TASK_ORDER:
            x_matrix, y_matrix = matrices[task]
            order = rng.permutation(y_matrix.shape[0])
            x_parts.append(x_matrix.ravel())
            y_parts.append(y_matrix[order].ravel())
        permutation[index] = pearson(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
        )

    pvalue = (
        np.sum(np.abs(permutation) >= abs(observed)) + 1
    ) / (permutation_iterations + 1)

    return {
        "rho_within_stratum": observed,
        "ci95_low_cluster_bootstrap": float(
            np.quantile(bootstrap[np.isfinite(bootstrap)], 0.025)
        ),
        "ci95_high_cluster_bootstrap": float(
            np.quantile(bootstrap[np.isfinite(bootstrap)], 0.975)
        ),
        "p_cluster_permutation": float(pvalue),
        "num_unique_queries": int(total_clusters),
        "num_reformulation_rows": int(len(clean)),
        "bootstrap_iterations": int(bootstrap_iterations),
        "permutation_iterations": int(permutation_iterations),
        "cluster_unit": "query_id within task",
        "strata": "generator × query_type × task",
    }


def choose_validation_sample(
    frame: pd.DataFrame,
    *,
    per_task: int,
    seed: int,
) -> pd.DataFrame:
    unique = frame[
        [
            "query_id",
            "task",
            "original_text",
            "file_paths",
            "path_anchor_tokens_v3",
            "entity_anchor_tokens_v3",
            "combined_anchor_tokens_v3",
        ]
    ].drop_duplicates("query_id")

    rng = np.random.default_rng(seed)
    pieces = []
    for task in TASK_ORDER:
        task_rows = unique[unique["task"].eq(task)].copy()
        positions = rng.choice(
            len(task_rows),
            size=min(per_task, len(task_rows)),
            replace=False,
        )
        pieces.append(task_rows.iloc[np.sort(positions)])

    sample = pd.concat(pieces, ignore_index=True)
    sample["manual_anchor_tokens"] = ""
    sample["proxy_correct_yes_no"] = ""
    sample["missing_anchor_tokens"] = ""
    sample["spurious_anchor_tokens"] = ""
    sample["reviewer_notes"] = ""
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--permutation-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--validation-per-task", type=int, default=10)
    args = parser.parse_args()

    root = args.root.resolve()
    input_path = require_first(
        [
            root
            / "outputs"
            / "tables"
            / "legalbench"
            / "anchor_preservation_per_query.csv",
            root / "anchor_preservation_per_query.csv",
        ]
    )

    output_dir = root / "outputs" / "anchor_phase2"
    figs_dir = root / "figs"
    tables_dir = root / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading and structurally auditing the existing proxy file...")
    frame = pd.read_csv(input_path)

    required = {
        "query_id", "task", "generator", "query_type",
        "original_text", "reformulation_text", "file_paths",
        *METRIC_DELTAS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    duplicate_keys = ["query_id", "generator", "query_type"]
    duplicate_count = int(frame.duplicated(duplicate_keys).sum())
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate query/method rows"
        )

    query_counts = frame.groupby("query_id").size()
    if not (query_counts == 6).all():
        raise ValueError(
            "Every query must have exactly six reformulations. "
            f"Observed counts: {query_counts.value_counts().to_dict()}"
        )

    if frame["query_id"].nunique() != 200 or len(frame) != 1200:
        raise ValueError(
            f"Expected 200 queries and 1,200 rows; observed "
            f"{frame['query_id'].nunique()} queries and {len(frame)} rows"
        )

    print("[2/6] Reconstructing source anchors present in the original query...")
    path_anchor_column = []
    entity_anchor_column = []
    combined_anchor_column = []

    kept_path_column = []
    dropped_path_column = []
    path_retention_column = []

    kept_entity_column = []
    dropped_entity_column = []
    entity_retention_column = []

    kept_combined_column = []
    dropped_combined_column = []
    combined_retention_column = []

    for row in frame.itertuples(index=False):
        path_anchors = path_overlap_anchor_tokens(
            row.original_text,
            row.file_paths,
        )
        entity_anchors = entity_anchor_tokens(row.original_text)
        combined_anchors = sorted(set(path_anchors) | set(entity_anchors))

        kept_path, dropped_path, path_retention = retained_tokens(
            path_anchors,
            row.reformulation_text,
        )
        kept_entity, dropped_entity, entity_retention = retained_tokens(
            entity_anchors,
            row.reformulation_text,
        )
        kept_combined, dropped_combined, combined_retention = retained_tokens(
            combined_anchors,
            row.reformulation_text,
        )

        path_anchor_column.append(", ".join(path_anchors))
        entity_anchor_column.append(", ".join(entity_anchors))
        combined_anchor_column.append(", ".join(combined_anchors))

        kept_path_column.append(", ".join(kept_path))
        dropped_path_column.append(", ".join(dropped_path))
        path_retention_column.append(path_retention)

        kept_entity_column.append(", ".join(kept_entity))
        dropped_entity_column.append(", ".join(dropped_entity))
        entity_retention_column.append(entity_retention)

        kept_combined_column.append(", ".join(kept_combined))
        dropped_combined_column.append(", ".join(dropped_combined))
        combined_retention_column.append(combined_retention)

    frame["path_anchor_tokens_v3"] = path_anchor_column
    frame["entity_anchor_tokens_v3"] = entity_anchor_column
    frame["combined_anchor_tokens_v3"] = combined_anchor_column
    frame["path_anchor_count_v3"] = [
        0 if not text else len(text.split(", ")) for text in path_anchor_column
    ]
    frame["entity_anchor_count_v3"] = [
        0 if not text else len(text.split(", ")) for text in entity_anchor_column
    ]
    frame["combined_anchor_count_v3"] = [
        0 if not text else len(text.split(", ")) for text in combined_anchor_column
    ]

    frame["kept_path_anchor_tokens_v3"] = kept_path_column
    frame["dropped_path_anchor_tokens_v3"] = dropped_path_column
    frame["path_anchor_retention_v3"] = path_retention_column

    frame["kept_entity_anchor_tokens_v3"] = kept_entity_column
    frame["dropped_entity_anchor_tokens_v3"] = dropped_entity_column
    frame["entity_anchor_retention_v3"] = entity_retention_column

    frame["kept_combined_anchor_tokens_v3"] = kept_combined_column
    frame["dropped_combined_anchor_tokens_v3"] = dropped_combined_column
    frame["combined_anchor_retention_v3"] = combined_retention_column

    # Quantify the conceptual problem in the old denominator.
    old_anchor_tokens_not_in_original = 0
    old_anchor_tokens_total = 0
    old_rows_with_impossible_tokens = 0

    if {"kept_anchor_tokens", "dropped_anchor_tokens"}.issubset(frame.columns):
        for row in frame.itertuples(index=False):
            old_tokens = []
            for field in (
                getattr(row, "kept_anchor_tokens", ""),
                getattr(row, "dropped_anchor_tokens", ""),
            ):
                if pd.notna(field):
                    old_tokens.extend(
                        token.strip()
                        for token in str(field).split(",")
                        if token.strip()
                    )

            original_set = set(simple_tokens(row.original_text))
            original_compact = re.sub(
                r"[^a-z0-9]+", "",
                ascii_lower(row.original_text),
            )
            impossible = [
                token for token in old_tokens
                if (
                    ascii_lower(token) not in original_set
                    and not (
                        len(ascii_lower(token)) >= 4
                        and ascii_lower(token) in original_compact
                    )
                )
            ]
            old_anchor_tokens_total += len(old_tokens)
            old_anchor_tokens_not_in_original += len(impossible)
            old_rows_with_impossible_tokens += int(bool(impossible))

    print("[3/6] Computing summaries and subgroup correlations...")
    summary = (
        frame.groupby(["generator", "query_type", "task"], sort=True)
        .agg(
            num_queries=("query_id", "nunique"),
            avg_path_anchor_count_v3=("path_anchor_count_v3", "mean"),
            avg_entity_anchor_count_v3=("entity_anchor_count_v3", "mean"),
            avg_combined_anchor_count_v3=("combined_anchor_count_v3", "mean"),
            avg_path_anchor_retention_v3=("path_anchor_retention_v3", "mean"),
            avg_entity_anchor_retention_v3=("entity_anchor_retention_v3", "mean"),
            avg_combined_anchor_retention_v3=(
                "combined_anchor_retention_v3", "mean"
            ),
            zero_combined_retention_rate=(
                "combined_anchor_retention_v3",
                lambda values: float((values == 0).mean()),
            ),
            delta_Recall_at_10=("delta_Recall@10", "mean"),
            delta_Recall_at_100=("delta_Recall@100", "mean"),
            delta_MRR_at_10=("delta_MRR@10", "mean"),
            delta_nDCG_at_10=("delta_nDCG@10", "mean"),
        )
        .reset_index()
    )

    subgroup_rows = []
    for (generator, query_type, task), group in frame.groupby(
        ["generator", "query_type", "task"],
        sort=True,
    ):
        x = group[PRIMARY_SIGNAL].to_numpy(dtype=float)
        y = group["delta_Recall@100"].to_numpy(dtype=float)
        rho = spearman(x, y)
        ci_low, ci_high = bootstrap_spearman_ci(
            x,
            y,
            iterations=args.bootstrap_iterations,
            seed=(
                args.seed
                + int(
                    hashlib.md5(
                        f"{generator}|{query_type}|{task}".encode()
                    ).hexdigest()[:8],
                    16,
                )
                % 100_000
            ),
        )
        subgroup_rows.append(
            {
                "generator": generator,
                "query_type": query_type,
                "task": task,
                "retention_signal": PRIMARY_SIGNAL,
                "metric_delta": "delta_Recall@100",
                "spearman_rho": rho,
                "ci95_low_query_bootstrap": ci_low,
                "ci95_high_query_bootstrap": ci_high,
                "p_raw_permutation": permutation_spearman_pvalue(
                    x,
                    y,
                    iterations=args.permutation_iterations,
                    seed=(
                        args.seed
                        + 500_000
                        + int(
                            hashlib.md5(
                                f"{generator}|{query_type}|{task}".encode()
                            ).hexdigest()[:8],
                            16,
                        )
                        % 100_000
                    ),
                ),
                "num_queries": int(group["query_id"].nunique()),
                "bootstrap_iterations": args.bootstrap_iterations,
                "permutation_iterations": args.permutation_iterations,
            }
        )

    subgroup = pd.DataFrame(subgroup_rows)
    subgroup["p_holm_24"] = holm_adjust(
        subgroup["p_raw_permutation"].tolist()
    )
    subgroup["significant_holm_0.05"] = subgroup["p_holm_24"].lt(0.05)

    print("[4/6] Computing query-clustered pooled correlations...")
    pooled_rows = []
    signals = (
        "combined_anchor_retention_v3",
        "path_anchor_retention_v3",
        "entity_anchor_retention_v3",
    )
    for signal in signals:
        for metric in METRIC_DELTAS:
            result = clustered_pooled_inference(
                frame,
                signal,
                metric,
                bootstrap_iterations=args.bootstrap_iterations,
                permutation_iterations=args.permutation_iterations,
                seed=(
                    args.seed
                    + int(
                        hashlib.md5(
                            f"{signal}|{metric}".encode()
                        ).hexdigest()[:8],
                        16,
                    )
                    % 100_000
                ),
            )
            result.update(
                {
                    "retention_signal": signal,
                    "metric_delta": metric,
                }
            )
            pooled_rows.append(result)

    pooled = pd.DataFrame(pooled_rows)
    pooled["p_holm_12"] = holm_adjust(
        pooled["p_cluster_permutation"].tolist()
    )
    pooled["significant_holm_0.05"] = pooled["p_holm_12"].lt(0.05)

    print("[5/6] Producing Figure 4 and the validation worksheet...")
    fig_frame = frame[
        frame[PRIMARY_SIGNAL].notna()
        & frame["delta_Recall@100"].notna()
    ].copy()

    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    marker_map = {"deepseek": "o", "gpt": "^"}

    for generator in GENERATORS:
        for query_type in QUERY_TYPES:
            rows = fig_frame[
                fig_frame["generator"].eq(generator)
                & fig_frame["query_type"].eq(query_type)
            ]
            label = (
                f"{generator.capitalize()} "
                f"{query_type.replace('_', ' ')}"
            )
            ax.scatter(
                rows[PRIMARY_SIGNAL],
                rows["delta_Recall@100"],
                alpha=0.36,
                s=22,
                marker=marker_map[generator],
                label=label,
            )

    # Descriptive binned trend, avoiding a causal/linear interpretation.
    bins = pd.cut(
        fig_frame[PRIMARY_SIGNAL],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    trend = (
        fig_frame.assign(_bin=bins)
        .groupby("_bin", observed=True)
        .agg(
            x=(PRIMARY_SIGNAL, "mean"),
            y=("delta_Recall@100", "mean"),
            n=("query_id", "size"),
        )
        .dropna()
    )
    ax.plot(
        trend["x"],
        trend["y"],
        linewidth=2,
        marker="s",
        label="Binned mean trend",
    )
    ax.axhline(0, linewidth=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_xlabel("Combined anchor-preservation proxy")
    ax.set_ylabel(r"$\Delta$Recall@100")
    ax.set_title(
        "Anchor preservation and Recall@100 change\n"
        "LegalBench-RAG mini direct replacement"
    )

    primary = pooled[
        pooled["retention_signal"].eq(PRIMARY_SIGNAL)
        & pooled["metric_delta"].eq("delta_Recall@100")
    ].iloc[0]
    annotation = (
        r"Within-stratum $\rho$="
        f"{primary['rho_within_stratum']:.3f}\n"
        f"cluster-permutation $p$="
        f"{primary['p_cluster_permutation']:.4g}"
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    ax.grid(alpha=0.22)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=2,
    )
    fig.subplots_adjust(bottom=0.28)

    fig_pdf = figs_dir / "figure4_anchor_preservation_recall100_v3.pdf"
    fig_png = figs_dir / "figure4_anchor_preservation_recall100_v3.png"
    fig.savefig(fig_pdf, bbox_inches="tight")
    fig.savefig(fig_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    validation = choose_validation_sample(
        frame,
        per_task=args.validation_per_task,
        seed=args.seed,
    )

    instructions = """# Manual validation of the anchor proxy

Review each unique query without looking at the retrieval delta.

For each row:

1. Read `original_text` and `file_paths`.
2. Write the source-identifying tokens or short phrases that are genuinely
   present in the original query in `manual_anchor_tokens`.
3. Mark `proxy_correct_yes_no` as `yes` only when the extracted combined
   anchors are an acceptable token-level approximation.
4. Record omitted anchors in `missing_anchor_tokens`.
5. Record generic or spurious extracted anchors in
   `spurious_anchor_tokens`.
6. Use `reviewer_notes` for ambiguous cases.

The worksheet is stratified by task with a deterministic seed. It must be
reviewed before the manuscript calls the score a validated APR. Until then,
use the expression “lexical anchor-preservation proxy”.
"""

    print("[6/6] Saving Phase 2B artifacts...")
    per_query_out = output_dir / "anchor_preservation_per_query_v3.csv"
    summary_out = output_dir / "anchor_proxy_summary_v3.csv"
    subgroup_out = output_dir / "anchor_proxy_subgroup_correlations_v3.csv"
    pooled_out = output_dir / "anchor_proxy_pooled_clustered_correlations_v3.csv"
    audit_out = output_dir / "anchor_proxy_structural_audit_v3.txt"
    manifest_out = output_dir / "anchor_proxy_input_manifest_v3.csv"

    frame.to_csv(per_query_out, index=False)
    summary.to_csv(summary_out, index=False)
    subgroup.to_csv(subgroup_out, index=False)
    pooled.to_csv(pooled_out, index=False)

    # Copies intended for direct manuscript/project consumption.
    summary.to_csv(tables_dir / "anchor_proxy_summary_v3.csv", index=False)
    subgroup.to_csv(
        tables_dir / "anchor_proxy_subgroup_correlations_v3.csv",
        index=False,
    )
    pooled.to_csv(
        tables_dir / "anchor_proxy_pooled_clustered_correlations_v3.csv",
        index=False,
    )
    validation.to_csv(
        tables_dir / "anchor_proxy_validation_sample_v3.csv",
        index=False,
    )
    (
        tables_dir / "anchor_proxy_validation_instructions_v3.md"
    ).write_text(instructions, encoding="utf-8")

    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    pd.DataFrame(
        [
            {
                "relative_path": str(input_path.relative_to(root)),
                "size_bytes": input_path.stat().st_size,
                "sha256": input_hash,
            }
        ]
    ).to_csv(manifest_out, index=False)

    old_zero_rows = (
        int(frame["anchor_token_count"].eq(0).sum())
        if "anchor_token_count" in frame.columns
        else -1
    )
    new_zero_rows = int(frame["combined_anchor_count_v3"].eq(0).sum())

    with audit_out.open("w", encoding="utf-8") as handle:
        handle.write("PHASE 2B — REFINED ANCHOR PROXY AUDIT\n")
        handle.write("================================\n\n")
        handle.write(f"Input: {input_path.relative_to(root)}\n")
        handle.write(f"Input SHA-256: {input_hash}\n")
        handle.write(f"Rows: {len(frame)}\n")
        handle.write(f"Unique queries: {frame['query_id'].nunique()}\n")
        handle.write("Reformulations per query: 6\n")
        handle.write(f"Duplicate query/method rows: {duplicate_count}\n\n")

        handle.write("OLD-PROXY DIAGNOSTIC\n")
        handle.write(
            "The original Phase 2A denominator contained filepath tokens that "
            "were absent from the original query.\n"
        )
        handle.write(
            f"Old anchor-token occurrences inspected: "
            f"{old_anchor_tokens_total}\n"
        )
        handle.write(
            f"Old token occurrences not found in the original query: "
            f"{old_anchor_tokens_not_in_original}\n"
        )
        handle.write(
            f"Rows containing at least one such token: "
            f"{old_rows_with_impossible_tokens}/{len(frame)}\n"
        )
        if old_zero_rows >= 0:
            handle.write(
                f"Rows with old anchor_token_count = 0: {old_zero_rows}\n"
            )
        handle.write(
            f"Rows with v3 combined_anchor_count = 0: {new_zero_rows}\n\n"
        )

        handle.write("PRIMARY POOLED RESULT\n")
        handle.write(
            f"Signal: {PRIMARY_SIGNAL}\n"
            f"Outcome: delta_Recall@100\n"
            f"Within-stratum rho: "
            f"{primary['rho_within_stratum']:.6f}\n"
            f"Cluster-bootstrap 95% CI: "
            f"[{primary['ci95_low_cluster_bootstrap']:.6f}, "
            f"{primary['ci95_high_cluster_bootstrap']:.6f}]\n"
            f"Cluster-permutation p: "
            f"{primary['p_cluster_permutation']:.8g}\n"
            f"Holm p across 12 pooled sensitivity tests: "
            f"{primary['p_holm_12']:.8g}\n"
            f"Bootstrap iterations: {args.bootstrap_iterations}\n"
            f"Permutation iterations: {args.permutation_iterations}\n\n"
        )

        handle.write("INTERPRETATION LIMIT\n")
        handle.write(
            "This remains a conservative lexical proxy. The new manual validation worksheet "
            "must be reviewed before calling it a validated APR.\n"
        )

    print("=" * 88)
    print("PHASE 2B COMPLETE")
    print("=" * 88)
    print(f"Rows analysed:                 {len(frame):,}")
    print(f"Unique query clusters:         {frame['query_id'].nunique():,}")
    print(
        "Old tokens absent from query: "
        f"{old_anchor_tokens_not_in_original:,}/"
        f"{old_anchor_tokens_total:,}"
    )
    print(f"Old zero-anchor rows:          {old_zero_rows:,}")
    print(f"V3 zero-anchor rows:           {new_zero_rows:,}")
    print(
        "Primary pooled rho:           "
        f"{primary['rho_within_stratum']:.4f}"
    )
    print(
        "Primary cluster p-value:       "
        f"{primary['p_cluster_permutation']:.6g}"
    )
    print("Saved:")
    for path in (
        per_query_out,
        summary_out,
        subgroup_out,
        pooled_out,
        audit_out,
        manifest_out,
        fig_pdf,
        fig_png,
        tables_dir / "anchor_proxy_validation_sample_v3.csv",
        tables_dir / "anchor_proxy_validation_instructions_v3.md",
    ):
        print(f"  {path.relative_to(root)}")


if __name__ == "__main__":
    main()
