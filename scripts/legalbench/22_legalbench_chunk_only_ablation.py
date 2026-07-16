"""Run the LegalBench chunk-only indexing ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


DATA_DIR = Path("data/processed/legalbench/rag_mini")
RUNS_DIR = Path("runs/legalbench/chunk_only")
OUT_DIR = Path("outputs/tables/legalbench/chunk_only")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
EPS = 1e-12


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def simple_tokenize(text: str) -> list[str]:
    text = str(text or "").lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def get_query_text(row: dict[str, Any]) -> str:
    for key in ["text", "original_text", "question", "query", "original"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_qrels(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )


def build_qrels_dict(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for qid, group in qrels_df.groupby("query_id"):
        qrels[str(qid)] = set(group.loc[group["relevance"] > 0, "doc_id"].astype(str))
    return qrels


def resolve_reformulation_path(generator: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Missing explicit {generator} reformulation file: {path}")
        return path

    base = DATA_DIR / "reformulations"
    search_dirs = [
        base / "normalized",
        base / "clean",
        base / "validated",
        base / "raw",
        base,
        DATA_DIR,
    ]
    patterns = [
        f"*{generator}*normalized*.jsonl",
        f"*{generator}*clean*.jsonl",
        f"*{generator}*valid*.jsonl",
        f"*{generator}*mini*.jsonl",
        f"*{generator}*.jsonl",
    ]

    candidates: list[Path] = []
    for directory in search_dirs:
        if not directory.exists():
            continue
        for pattern in patterns:
            candidates.extend(sorted(directory.glob(pattern)))

    # Deduplicate while preserving priority order.
    seen = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    if not unique:
        searched = "\n".join(str(d / p) for d in search_dirs for p in patterns)
        raise FileNotFoundError(
            f"Could not auto-detect {generator} reformulation JSONL.\n"
            f"Pass it explicitly with --{generator}-reformulations.\nSearched:\n{searched}"
        )

    return unique[0]


def load_reformulations(path: Path, generator: str) -> dict[str, dict[str, str]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, str]] = {}
    required = ["legal_rewrite", "keyword_expansion", "hyde_style"]

    for row in rows:
        qid = str(row.get("query_id", "")).strip()
        if not qid:
            continue

        values: dict[str, str] = {}
        for field in required:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Missing/empty {field} for {generator} query_id={qid} in {path}")
            values[field] = value.strip()

        out[qid] = values

    if not out:
        raise ValueError(f"No valid reformulations loaded from {path}")

    return out


def evaluate_run_per_query(
    run_df: pd.DataFrame,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
    method: str,
) -> pd.DataFrame:
    grouped = run_df.groupby("query_id", sort=False)
    rows = []

    for qid, relevant_docs in qrels.items():
        if qid in grouped.groups:
            ranked_docs = (
                grouped.get_group(qid)
                .sort_values("rank")["doc_id"]
                .astype(str)
                .tolist()
            )
        else:
            ranked_docs = []

        n_relevant = len(relevant_docs)
        top10 = set(ranked_docs[:10])
        top100 = set(ranked_docs[:100])

        recall10 = len(top10 & relevant_docs) / n_relevant if n_relevant else 0.0
        recall100 = len(top100 & relevant_docs) / n_relevant if n_relevant else 0.0

        mrr10 = 0.0
        first_relevant_rank = None
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                mrr10 = 1.0 / rank
                first_relevant_rank = rank
                break

        dcg10 = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg10 += 1.0 / math.log2(rank + 1)

        idcg10 = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0

        rows.append(
            {
                "method": method,
                "query_id": qid,
                "task": query_task.get(qid, "unknown"),
                "num_relevant": n_relevant,
                "Recall@10": recall10,
                "Recall@100": recall100,
                "MRR@10": mrr10,
                "nDCG@10": ndcg10,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(per_query_df: pd.DataFrame) -> dict[str, float]:
    return {metric: float(per_query_df[metric].mean()) for metric in METRICS}


def retrieve_bm25_view(
    *,
    method: str,
    query_texts: dict[str, str],
    bm25: BM25Okapi,
    doc_ids: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    top_k = min(top_k, len(doc_ids))

    for qid, text in tqdm(query_texts.items(), desc=f"Retrieving {method}"):
        scores = bm25.get_scores(simple_tokenize(text))

        if top_k < len(scores):
            candidate_indices = np.argpartition(-scores, top_k - 1)[:top_k]
        else:
            candidate_indices = np.arange(len(scores))

        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (-float(scores[idx]), doc_ids[idx]),
        )

        for rank, idx in enumerate(ranked_indices, start=1):
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                    "method": method,
                }
            )

    return pd.DataFrame(rows)


def rrf_fuse(
    *,
    method: str,
    run_dfs: list[pd.DataFrame],
    rrf_k: int,
    top_k: int,
) -> pd.DataFrame:
    scores_by_query: dict[str, dict[str, float]] = {}

    for run_df in run_dfs:
        for row in run_df[["query_id", "doc_id", "rank"]].itertuples(index=False):
            qid = str(row.query_id)
            doc_id = str(row.doc_id)
            rank = int(row.rank)
            scores_by_query.setdefault(qid, {})[doc_id] = (
                scores_by_query.setdefault(qid, {}).get(doc_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    fused_rows = []
    for qid, doc_scores in scores_by_query.items():
        ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            fused_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": float(score),
                    "method": method,
                }
            )

    return pd.DataFrame(fused_rows)


def save_run(run_df: pd.DataFrame, method: str) -> Path:
    path = RUNS_DIR / f"{method}.tsv"
    run_df.to_csv(path, sep="\t", index=False)
    return path


def sign_flip_pvalue(deltas: np.ndarray, iterations: int, seed: int) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = float(np.mean(deltas))

    if np.all(np.abs(deltas) <= EPS):
        return 1.0

    rng = np.random.default_rng(seed)
    # Two-sided paired sign-flip test.
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(deltas)))
    permuted = signs * deltas[None, :]
    perm_means = permuted.mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1.0) / (iterations + 1.0)
    return float(min(max(p, 0.0), 1.0))


def holm_adjust(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    """Holm adjustment over all provided p-values."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted_raw: list[tuple[tuple[str, str], float]] = []

    for i, (key, p) in enumerate(items):
        adjusted_raw.append((key, min((m - i) * p, 1.0)))

    # Enforce monotonicity in sorted order.
    out: dict[tuple[str, str], float] = {}
    running = 0.0
    for key, adj in adjusted_raw:
        running = max(running, adj)
        out[key] = min(running, 1.0)

    return out


def build_candidate_summary(
    *,
    per_query_all: pd.DataFrame,
    baseline_method: str,
    method_meta: dict[str, dict[str, str]],
    stats_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = per_query_all[per_query_all["method"] == baseline_method][
        ["query_id", *METRICS]
    ].copy()
    baseline = baseline.rename(columns={m: f"baseline_{m}" for m in METRICS})

    gains_rows = []
    pvalues: dict[tuple[str, str], float] = {}

    for method in method_meta:
        cur = per_query_all[per_query_all["method"] == method].copy()
        merged = cur.merge(baseline, on="query_id", how="left")
        if merged[[f"baseline_{m}" for m in METRICS]].isna().any().any():
            raise ValueError(f"Missing baseline metrics while comparing {method}")

        for m in METRICS:
            merged[f"Delta {m}"] = merged[m] - merged[f"baseline_{m}"]

        gains_rows.append(merged)

        if method != baseline_method and stats_iterations > 0:
            for j, m in enumerate(METRICS):
                key = (method, m)
                pvalues[key] = sign_flip_pvalue(
                    merged[f"Delta {m}"].to_numpy(),
                    iterations=stats_iterations,
                    seed=seed + 1000 * (j + 1) + abs(hash(method)) % 997,
                )

    gains_df = pd.concat(gains_rows, ignore_index=True)
    p_holm = holm_adjust(pvalues) if pvalues else {}

    summary_rows = []
    for method, meta in method_meta.items():
        df = gains_df[gains_df["method"] == method]
        row: dict[str, Any] = {
            "method": method,
            "family": meta.get("family", ""),
            "setting": meta.get("setting", ""),
            "index_mode": "chunk_only",
            "num_queries": int(len(df)),
        }
        sig_pos = []
        sig_neg = []
        for m in METRICS:
            value = float(df[m].mean())
            delta = 0.0 if method == baseline_method else float(df[f"Delta {m}"].mean())
            harm = 0.0 if method == baseline_method else float((df[f"Delta {m}"] < -EPS).mean() * 100.0)
            improve = 0.0 if method == baseline_method else float((df[f"Delta {m}"] > EPS).mean() * 100.0)
            neutral = 100.0 if method == baseline_method else float((df[f"Delta {m}"].abs() <= EPS).mean() * 100.0)

            row[m] = value
            row[f"Delta {m}"] = delta
            row[f"Harm {m} %"] = harm
            row[f"Improve {m} %"] = improve
            row[f"Neutral {m} %"] = neutral

            if method == baseline_method:
                row[f"pRaw {m}"] = ""
                row[f"pHolm {m}"] = ""
            else:
                raw = pvalues.get((method, m), float("nan"))
                adj = p_holm.get((method, m), float("nan"))
                row[f"pRaw {m}"] = raw
                row[f"pHolm {m}"] = adj
                if not math.isnan(adj) and adj < 0.05:
                    if delta > 0:
                        sig_pos.append(m)
                    elif delta < 0:
                        sig_neg.append(m)

        row["Holm significant positive metrics"] = ", ".join(sig_pos)
        row["Holm significant negative metrics"] = ", ".join(sig_neg)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, gains_df


def build_by_task(per_query_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, task), group in per_query_all.groupby(["method", "task"], sort=False):
        row = {"method": method, "task": task, "num_queries": int(len(group))}
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LegalBench-RAG mini chunk-only BM25/RRF ablation."
    )
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--stats-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deepseek-reformulations", default=None)
    parser.add_argument("--gpt-reformulations", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.tsv"

    for path in [corpus_path, queries_path, qrels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    deepseek_path = resolve_reformulation_path("deepseek", args.deepseek_reformulations)
    gpt_path = resolve_reformulation_path("gpt", args.gpt_reformulations)

    print("=" * 90)
    print("LegalBench-RAG mini — CHUNK-ONLY ablation")
    print("=" * 90)
    print(f"Corpus:   {corpus_path}")
    print(f"Queries:  {queries_path}")
    print(f"Qrels:    {qrels_path}")
    print(f"DeepSeek: {deepseek_path}")
    print(f"GPT:      {gpt_path}")

    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)
    deepseek = load_reformulations(deepseek_path, "deepseek")
    gpt = load_reformulations(gpt_path, "gpt")

    doc_ids = [str(doc["doc_id"]) for doc in corpus]
    # CHUNK-ONLY: on indexe strictement le texte du chunk, sans file_path ni filename.
    doc_texts = [str(doc.get("text", "") or "") for doc in corpus]

    original_queries = {str(row["query_id"]): get_query_text(row) for row in queries}
    query_task = {str(row["query_id"]): str(row.get("task", "unknown")) for row in queries}

    qids = sorted(qrels.keys())
    missing_original = [qid for qid in qids if not original_queries.get(qid)]
    missing_deepseek = [qid for qid in qids if qid not in deepseek]
    missing_gpt = [qid for qid in qids if qid not in gpt]
    if missing_original or missing_deepseek or missing_gpt:
        raise ValueError(
            "Missing query/reformulation data:\n"
            f"  original missing: {missing_original[:5]} total={len(missing_original)}\n"
            f"  deepseek missing: {missing_deepseek[:5]} total={len(missing_deepseek)}\n"
            f"  gpt missing: {missing_gpt[:5]} total={len(missing_gpt)}"
        )

    # Keep only qrels query ids, in stable order.
    original_queries = {qid: original_queries[qid] for qid in qids}
    ds_keyword_queries = {qid: deepseek[qid]["keyword_expansion"] for qid in qids}
    ds_legal_queries = {qid: deepseek[qid]["legal_rewrite"] for qid in qids}
    gpt_keyword_queries = {qid: gpt[qid]["keyword_expansion"] for qid in qids}
    gpt_legal_queries = {qid: gpt[qid]["legal_rewrite"] for qid in qids}
    gpt_hyde_queries = {qid: gpt[qid]["hyde_style"] for qid in qids}

    print(f"Corpus chunks: {len(corpus)}")
    print(f"Queries/qrels: {len(qids)}")
    print(f"Qrels rows:    {len(qrels_df)}")

    print("\nTokenizing CHUNK-ONLY corpus...")
    tokenized_corpus = [simple_tokenize(text) for text in tqdm(doc_texts, desc="Tokenizing")]

    print("\nBuilding BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    # Run all atomic views needed by the requested configurations.
    atomic_query_sets = {
        "lb_chunk_original_bm25": original_queries,
        "lb_chunk_deepseek_keyword_replacement": ds_keyword_queries,
        "lb_chunk_deepseek_legal_rewrite_view": ds_legal_queries,
        "lb_chunk_gpt_keyword_replacement": gpt_keyword_queries,
        "lb_chunk_gpt_legal_rewrite_view": gpt_legal_queries,
        "lb_chunk_gpt_hyde_replacement": gpt_hyde_queries,
    }

    atomic_runs: dict[str, pd.DataFrame] = {}
    for method, query_texts in atomic_query_sets.items():
        run_df = retrieve_bm25_view(
            method=method,
            query_texts=query_texts,
            bm25=bm25,
            doc_ids=doc_ids,
            top_k=args.top_k,
        )
        atomic_runs[method] = run_df
        path = save_run(run_df, method)
        print(f"Saved run: {path}")

    # Requested RRF configurations.
    rrf_ds_kw_gpt_legal = rrf_fuse(
        method="lb_chunk_rrf_original_deepseek_keyword_gpt_legal",
        run_dfs=[
            atomic_runs["lb_chunk_original_bm25"],
            atomic_runs["lb_chunk_deepseek_keyword_replacement"],
            atomic_runs["lb_chunk_gpt_legal_rewrite_view"],
        ],
        rrf_k=args.rrf_k,
        top_k=args.top_k,
    )
    save_run(rrf_ds_kw_gpt_legal, "lb_chunk_rrf_original_deepseek_keyword_gpt_legal")

    rrf_all_nonhyde = rrf_fuse(
        method="lb_chunk_rrf_original_all_nonhyde",
        run_dfs=[
            atomic_runs["lb_chunk_original_bm25"],
            atomic_runs["lb_chunk_deepseek_legal_rewrite_view"],
            atomic_runs["lb_chunk_deepseek_keyword_replacement"],
            atomic_runs["lb_chunk_gpt_legal_rewrite_view"],
            atomic_runs["lb_chunk_gpt_keyword_replacement"],
        ],
        rrf_k=args.rrf_k,
        top_k=args.top_k,
    )
    save_run(rrf_all_nonhyde, "lb_chunk_rrf_original_all_nonhyde")

    # Evaluate only requested final methods, not helper legal_rewrite atomic views.
    final_runs = {
        "lb_chunk_original_bm25": atomic_runs["lb_chunk_original_bm25"],
        "lb_chunk_deepseek_keyword_replacement": atomic_runs["lb_chunk_deepseek_keyword_replacement"],
        "lb_chunk_gpt_keyword_replacement": atomic_runs["lb_chunk_gpt_keyword_replacement"],
        "lb_chunk_gpt_hyde_replacement": atomic_runs["lb_chunk_gpt_hyde_replacement"],
        "lb_chunk_rrf_original_deepseek_keyword_gpt_legal": rrf_ds_kw_gpt_legal,
        "lb_chunk_rrf_original_all_nonhyde": rrf_all_nonhyde,
    }

    method_meta = {
        "lb_chunk_original_bm25": {
            "family": "BM25 original",
            "setting": "chunk-only original query",
        },
        "lb_chunk_deepseek_keyword_replacement": {
            "family": "Replacement",
            "setting": "DeepSeek keyword, chunk-only",
        },
        "lb_chunk_gpt_keyword_replacement": {
            "family": "Replacement",
            "setting": "GPT keyword, chunk-only",
        },
        "lb_chunk_gpt_hyde_replacement": {
            "family": "Replacement",
            "setting": "GPT HyDE, chunk-only",
        },
        "lb_chunk_rrf_original_deepseek_keyword_gpt_legal": {
            "family": "RRF multi-view",
            "setting": "original + DeepSeek keyword + GPT legal, chunk-only",
        },
        "lb_chunk_rrf_original_all_nonhyde": {
            "family": "RRF multi-view",
            "setting": "original + DS legal + DS keyword + GPT legal + GPT keyword, chunk-only",
        },
    }

    per_query_frames = []
    for method, run_df in final_runs.items():
        per_query = evaluate_run_per_query(
            run_df=run_df,
            qrels=qrels,
            query_task=query_task,
            method=method,
        )
        per_query_frames.append(per_query)

    per_query_all = pd.concat(per_query_frames, ignore_index=True)

    summary_df, gains_df = build_candidate_summary(
        per_query_all=per_query_all,
        baseline_method="lb_chunk_original_bm25",
        method_meta=method_meta,
        stats_iterations=args.stats_iterations,
        seed=args.seed,
    )

    by_task_df = build_by_task(per_query_all)

    summary_path = OUT_DIR / "legalbench_chunk_only_ablation_summary.csv"
    gains_path = OUT_DIR / "legalbench_chunk_only_ablation_per_query.csv"
    by_task_path = OUT_DIR / "legalbench_chunk_only_ablation_by_task.csv"
    summary_md_path = OUT_DIR / "legalbench_chunk_only_ablation_summary.md"

    summary_df.to_csv(summary_path, index=False)
    gains_df.to_csv(gains_path, index=False)
    by_task_df.to_csv(by_task_path, index=False)
    summary_df.to_markdown(summary_md_path, index=False)

    print("\n" + "=" * 90)
    print("CHUNK-ONLY ABLATION SUMMARY")
    print("=" * 90)
    display_cols = [
        "method",
        "family",
        "Recall@10",
        "Delta Recall@10",
        "pHolm Recall@10",
        "Recall@100",
        "Delta Recall@100",
        "Harm Recall@100 %",
        "pHolm Recall@100",
        "MRR@10",
        "Delta MRR@10",
        "pHolm MRR@10",
        "nDCG@10",
        "Delta nDCG@10",
        "Harm nDCG@10 %",
        "pHolm nDCG@10",
        "Holm significant positive metrics",
        "Holm significant negative metrics",
    ]
    print(summary_df[display_cols].to_string(index=False))

    print("\nSaved outputs:")
    print(summary_path)
    print(gains_path)
    print(by_task_path)
    print(summary_md_path)
    print(f"Runs dir: {RUNS_DIR}")


if __name__ == "__main__":
    main()
