"""Legacy dense and hybrid LegalBench implementation retained for traceability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Missing dependency: sentence-transformers. Install with:\n"
        "pip install sentence-transformers torch"
    ) from exc


DATA_DIR = Path("data/processed/legalbench/rag_mini")
RUNS_DIR = Path("runs/legalbench/dense_hybrid")
OUT_DIR = Path("outputs/tables/legalbench/dense_hybrid")
CACHE_DIR = Path("outputs/cache/legalbench_dense_hybrid")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
EPS = 1e-12


def stable_int_hash(text: str, modulo: int | None = None) -> int:
    """Return a deterministic integer hash.

    Python's built-in hash() is intentionally randomised between
    processes, which would make permutation-test seeds vary from
    one run to another. This helper keeps statistical results
    reproducible for a fixed --seed.
    """
    value = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
    return value % modulo if modulo else value


def float_tag(value: float) -> str:
    """Make a filesystem-safe representation of a float."""
    return str(value).replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class ModelSpec:
    short_name: str
    hf_name: str
    query_prefix: str
    doc_prefix: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "e5": ModelSpec(
        short_name="e5",
        hf_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "bge": ModelSpec(
        short_name="bge",
        hf_name="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
    ),
}


# ----------------------------
# Loading utilities
# ----------------------------

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


def get_file_path(row: dict[str, Any]) -> str:
    for key in ["file_path", "filepath", "path", "source", "resolved_file_path"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.replace("\\", "/").strip()
    return ""


def make_doc_text(row: dict[str, Any], index_mode: str) -> str:
    chunk = str(row.get("text", "") or "")
    file_path = get_file_path(row)
    filename = Path(file_path).name if file_path else ""

    if index_mode == "chunk_only":
        return chunk
    if index_mode == "filename_chunk":
        return f"{filename}\n{chunk}".strip()
    if index_mode == "filepath_chunk":
        return f"{file_path}\n{chunk}".strip()
    raise ValueError(f"Unknown index_mode: {index_mode}")


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
    search_dirs = [base / "normalized", base / "clean", base / "validated", base / "raw", base, DATA_DIR]
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

    seen = set()
    unique: list[Path] = []
    for path in candidates:
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    if not unique:
        raise FileNotFoundError(
            f"Could not auto-detect {generator} reformulation JSONL. "
            f"Pass it explicitly with --{generator}-reformulations."
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


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_run_per_query(
    run_df: pd.DataFrame,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
    method: str,
    group_name: str,
    baseline_method: str,
    model_name: str,
    retriever_family: str,
    index_mode: str,
) -> pd.DataFrame:
    grouped = run_df.groupby("query_id", sort=False)
    rows = []

    for qid, relevant_docs in qrels.items():
        if qid in grouped.groups:
            ranked_docs = grouped.get_group(qid).sort_values("rank")["doc_id"].astype(str).tolist()
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

        idcg10 = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(n_relevant, 10) + 1))
        ndcg10 = dcg10 / idcg10 if idcg10 > 0 else 0.0

        rows.append(
            {
                "method": method,
                "group": group_name,
                "baseline_method": baseline_method,
                "model": model_name,
                "retriever_family": retriever_family,
                "index_mode": index_mode,
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


def sign_flip_pvalue(deltas: np.ndarray, iterations: int, seed: int) -> float:
    deltas = np.asarray(deltas, dtype=float)
    observed = float(np.mean(deltas))
    if np.all(np.abs(deltas) <= EPS):
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(iterations, len(deltas)))
    perm_means = (signs * deltas[None, :]).mean(axis=1)
    p = (np.sum(np.abs(perm_means) >= abs(observed)) + 1.0) / (iterations + 1.0)
    return float(min(max(p, 0.0), 1.0))


def holm_adjust(pvalues: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted_raw: list[tuple[tuple[str, str], float]] = []
    for i, (key, p) in enumerate(items):
        adjusted_raw.append((key, min((m - i) * p, 1.0)))

    out: dict[tuple[str, str], float] = {}
    running = 0.0
    for key, adj in adjusted_raw:
        running = max(running, adj)
        out[key] = min(running, 1.0)
    return out


def build_candidate_summary(per_query_all: pd.DataFrame, stats_iterations: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    gains_all = []
    summary_rows = []

    for group_name, group_df in per_query_all.groupby("group", sort=False):
        baseline_method = str(group_df["baseline_method"].iloc[0])
        baseline = group_df[group_df["method"] == baseline_method][["query_id", *METRICS]].copy()
        baseline = baseline.rename(columns={m: f"baseline_{m}" for m in METRICS})

        group_gains = []
        pvalues: dict[tuple[str, str], float] = {}

        for method, method_df in group_df.groupby("method", sort=False):
            merged = method_df.merge(baseline, on="query_id", how="left")
            if merged[[f"baseline_{m}" for m in METRICS]].isna().any().any():
                raise ValueError(f"Missing baseline metrics while comparing {method}")
            for m in METRICS:
                merged[f"Delta {m}"] = merged[m] - merged[f"baseline_{m}"]
            group_gains.append(merged)

            if method != baseline_method and stats_iterations > 0:
                for j, m in enumerate(METRICS):
                    pvalues[(method, m)] = sign_flip_pvalue(
                        merged[f"Delta {m}"].to_numpy(),
                        iterations=stats_iterations,
                        seed=seed + 1000 * (j + 1) + stable_int_hash(method, modulo=997),
                    )

        p_holm = holm_adjust(pvalues) if pvalues else {}
        group_gains_df = pd.concat(group_gains, ignore_index=True)
        gains_all.append(group_gains_df)

        for method, df in group_gains_df.groupby("method", sort=False):
            row: dict[str, Any] = {
                "method": method,
                "group": group_name,
                "baseline_method": baseline_method,
                "model": str(df["model"].iloc[0]),
                "retriever_family": str(df["retriever_family"].iloc[0]),
                "index_mode": str(df["index_mode"].iloc[0]),
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

    return pd.DataFrame(summary_rows), pd.concat(gains_all, ignore_index=True)


def build_by_task(per_query_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, task), group in per_query_all.groupby(["method", "task"], sort=False):
        row = {
            "method": method,
            "group": str(group["group"].iloc[0]),
            "model": str(group["model"].iloc[0]),
            "retriever_family": str(group["retriever_family"].iloc[0]),
            "index_mode": str(group["index_mode"].iloc[0]),
            "task": task,
            "num_queries": int(len(group)),
        }
        for metric in METRICS:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------
# Retrieval
# ----------------------------

def save_run(run_df: pd.DataFrame, method: str) -> Path:
    path = RUNS_DIR / f"{method}.tsv"
    run_df.to_csv(path, sep="\t", index=False)
    return path


def dense_retrieve_from_embeddings(
    *,
    method: str,
    query_ids: list[str],
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    doc_ids: list[str],
    top_k: int,
    query_batch_size: int,
) -> pd.DataFrame:
    rows = []
    top_k = min(top_k, len(doc_ids))
    doc_emb_t = doc_emb.T

    for start in tqdm(range(0, len(query_ids), query_batch_size), desc=f"Dense retrieving {method}"):
        end = min(start + query_batch_size, len(query_ids))
        qids_batch = query_ids[start:end]
        q_emb_batch = query_emb[start:end]
        scores = np.matmul(q_emb_batch, doc_emb_t)  # cosine if embeddings normalized

        for local_i, qid in enumerate(qids_batch):
            row_scores = scores[local_i]
            if top_k < len(row_scores):
                idx = np.argpartition(-row_scores, top_k - 1)[:top_k]
            else:
                idx = np.arange(len(row_scores))
            ranked = sorted(idx, key=lambda j: (-float(row_scores[j]), doc_ids[j]))
            for rank, j in enumerate(ranked, start=1):
                rows.append({"query_id": qid, "doc_id": doc_ids[j], "rank": rank, "score": float(row_scores[j]), "method": method})

    return pd.DataFrame(rows)


def bm25_retrieve(
    *,
    method: str,
    query_texts: dict[str, str],
    bm25: BM25Okapi,
    doc_ids: list[str],
    top_k: int,
) -> pd.DataFrame:
    rows = []
    top_k = min(top_k, len(doc_ids))
    for qid, text in tqdm(query_texts.items(), desc=f"BM25 retrieving {method}"):
        scores = bm25.get_scores(simple_tokenize(text))
        if top_k < len(scores):
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
        else:
            idx = np.arange(len(scores))
        ranked = sorted(idx, key=lambda j: (-float(scores[j]), doc_ids[j]))
        for rank, j in enumerate(ranked, start=1):
            rows.append({"query_id": qid, "doc_id": doc_ids[j], "rank": rank, "score": float(scores[j]), "method": method})
    return pd.DataFrame(rows)


def minmax_dict(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vals = np.array(list(values.values()), dtype=float)
    lo = float(vals.min())
    hi = float(vals.max())
    if hi - lo <= EPS:
        return {k: 0.0 for k in values}
    return {k: (float(v) - lo) / (hi - lo) for k, v in values.items()}


def hybrid_fuse_scores(
    *,
    method: str,
    bm25_run: pd.DataFrame,
    dense_run: pd.DataFrame,
    dense_weight: float,
    top_k: int,
) -> pd.DataFrame:
    rows = []
    all_qids = sorted(set(bm25_run["query_id"].astype(str)) | set(dense_run["query_id"].astype(str)))
    bm25_groups = dict(tuple(bm25_run.groupby("query_id", sort=False)))
    dense_groups = dict(tuple(dense_run.groupby("query_id", sort=False)))

    for qid in tqdm(all_qids, desc=f"Hybrid fusing {method}"):
        bm25_scores = {}
        dense_scores = {}
        if qid in bm25_groups:
            bm25_scores = {str(r.doc_id): float(r.score) for r in bm25_groups[qid][["doc_id", "score"]].itertuples(index=False)}
        if qid in dense_groups:
            dense_scores = {str(r.doc_id): float(r.score) for r in dense_groups[qid][["doc_id", "score"]].itertuples(index=False)}
        docs = sorted(set(bm25_scores) | set(dense_scores))
        b_norm = minmax_dict({d: bm25_scores.get(d, 0.0) for d in docs})
        d_norm = minmax_dict({d: dense_scores.get(d, 0.0) for d in docs})
        scores = {d: (1.0 - dense_weight) * b_norm.get(d, 0.0) + dense_weight * d_norm.get(d, 0.0) for d in docs}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append({"query_id": qid, "doc_id": doc_id, "rank": rank, "score": float(score), "method": method})
    return pd.DataFrame(rows)


def rrf_fuse(method: str, run_dfs: list[pd.DataFrame], rrf_k: int, top_k: int) -> pd.DataFrame:
    scores_by_query: dict[str, dict[str, float]] = {}
    for run_df in run_dfs:
        for row in run_df[["query_id", "doc_id", "rank"]].itertuples(index=False):
            qid = str(row.query_id)
            doc_id = str(row.doc_id)
            rank = int(row.rank)
            scores_by_query.setdefault(qid, {})[doc_id] = scores_by_query.setdefault(qid, {}).get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

    rows = []
    for qid, doc_scores in scores_by_query.items():
        ranked = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        for rank, (doc_id, score) in enumerate(ranked, start=1):
            rows.append({"query_id": qid, "doc_id": doc_id, "rank": rank, "score": float(score), "method": method})
    return pd.DataFrame(rows)


# ----------------------------
# Embeddings
# ----------------------------

def safe_cache_name(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def encode_texts_cached(
    *,
    model: SentenceTransformer,
    texts: list[str],
    prefix: str,
    cache_path: Path,
    batch_size: int,
    device: str | None,
    force: bool,
    description: str,
) -> np.ndarray:
    if cache_path.exists() and not force:
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)

    encoded_texts = [prefix + str(t or "") for t in texts]
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "show_progress_bar": True,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
    }
    if device and device != "auto":
        kwargs["device"] = device

    print(f"Encoding {description} ({len(encoded_texts)} texts) -> {cache_path}")
    emb = model.encode(encoded_texts, **kwargs).astype("float32")
    np.save(cache_path, emb)
    return emb


# ----------------------------
# Main pipeline
# ----------------------------

def build_query_views(
    qids: list[str],
    original_queries: dict[str, str],
    deepseek: dict[str, dict[str, str]],
    gpt: dict[str, dict[str, str]],
) -> dict[str, dict[str, str]]:
    views: dict[str, dict[str, str]] = {
        "original": {qid: original_queries[qid] for qid in qids},
        "deepseek_legal": {qid: deepseek[qid]["legal_rewrite"] for qid in qids},
        "deepseek_keyword": {qid: deepseek[qid]["keyword_expansion"] for qid in qids},
        "deepseek_hyde": {qid: deepseek[qid]["hyde_style"] for qid in qids},
        "gpt_legal": {qid: gpt[qid]["legal_rewrite"] for qid in qids},
        "gpt_keyword": {qid: gpt[qid]["keyword_expansion"] for qid in qids},
        "gpt_hyde": {qid: gpt[qid]["hyde_style"] for qid in qids},
    }
    return views


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LegalBench-RAG mini dense and hybrid baselines.")
    parser.add_argument("--models", default="e5,bge", help="Comma-separated list among: e5,bge")
    parser.add_argument("--index-mode", choices=["filepath_chunk", "filename_chunk", "chunk_only"], default="filepath_chunk")
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Weight of dense score in hybrid score-level fusion")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--stats-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-recompute-embeddings", action="store_true")
    parser.add_argument("--deepseek-reformulations", default=None)
    parser.add_argument("--gpt-reformulations", default=None)
    parser.add_argument("--max-docs", type=int, default=None, help="Debug only: truncate corpus")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not (0.0 <= args.dense_weight <= 1.0):
        raise ValueError("--dense-weight must be between 0 and 1")

    model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    for key in model_keys:
        if key not in MODEL_SPECS:
            raise ValueError(f"Unknown model key {key}. Available: {sorted(MODEL_SPECS)}")

    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.tsv"
    for path in [corpus_path, queries_path, qrels_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    deepseek_path = resolve_reformulation_path("deepseek", args.deepseek_reformulations)
    gpt_path = resolve_reformulation_path("gpt", args.gpt_reformulations)

    print("=" * 90)
    print("LegalBench-RAG mini — DENSE + HYBRID baselines")
    print("=" * 90)
    print(f"Index mode:    {args.index_mode}")
    print(f"Models:        {model_keys}")
    print(f"Dense weight:  {args.dense_weight}")
    print("Dense search:   exact cosine over normalised embeddings (no FAISS)")
    print(f"Corpus:        {corpus_path}")
    print(f"Queries:       {queries_path}")
    print(f"Qrels:         {qrels_path}")
    print(f"DeepSeek:      {deepseek_path}")
    print(f"GPT:           {gpt_path}")

    corpus = read_jsonl(corpus_path)
    if args.max_docs is not None:
        corpus = corpus[: args.max_docs]
        print(f"DEBUG: corpus truncated to {len(corpus)} docs")
    queries = read_jsonl(queries_path)
    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)
    deepseek = load_reformulations(deepseek_path, "deepseek")
    gpt = load_reformulations(gpt_path, "gpt")

    doc_ids = [str(doc["doc_id"]) for doc in corpus]
    doc_texts = [make_doc_text(doc, args.index_mode) for doc in corpus]

    original_queries = {str(row["query_id"]): get_query_text(row) for row in queries}
    query_task = {str(row["query_id"]): str(row.get("task", "unknown")) for row in queries}

    # In max-docs debug mode, keep all qrels but metrics may be artificially low.
    qids = sorted(qrels.keys())
    missing = [qid for qid in qids if not original_queries.get(qid) or qid not in deepseek or qid not in gpt]
    if missing:
        raise ValueError(f"Missing original or reformulations for query ids: {missing[:10]} ...")

    query_views = build_query_views(qids, original_queries, deepseek, gpt)
    replacement_views = [
        "deepseek_legal",
        "deepseek_keyword",
        "deepseek_hyde",
        "gpt_legal",
        "gpt_keyword",
        "gpt_hyde",
    ]
    all_nonhyde_views = ["deepseek_legal", "deepseek_keyword", "gpt_legal", "gpt_keyword"]

    # BM25 index is needed for hybrid runs.
    print("Building BM25 index for hybrid scoring...")
    bm25 = BM25Okapi([simple_tokenize(t) for t in tqdm(doc_texts, desc="Tokenizing corpus")])

    dense_weight_tag = float_tag(args.dense_weight)
    models_tag = "-".join(model_keys)
    output_suffix = f"{args.index_mode}_models-{models_tag}_dw{dense_weight_tag}"

    all_per_query: list[pd.DataFrame] = []

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        print("\n" + "=" * 90)
        print(f"Model: {spec.short_name} — {spec.hf_name}")
        print("=" * 90)
        model = SentenceTransformer(spec.hf_name, device=None if args.device == "auto" else args.device)

        cache_base = safe_cache_name(f"{spec.short_name}_{args.index_mode}_{len(doc_ids)}docs")
        doc_cache = CACHE_DIR / f"doc_{cache_base}.npy"
        doc_emb = encode_texts_cached(
            model=model,
            texts=doc_texts,
            prefix=spec.doc_prefix,
            cache_path=doc_cache,
            batch_size=args.batch_size,
            device=args.device,
            force=args.force_recompute_embeddings,
            description=f"documents for {spec.short_name}",
        )

        # Retrieve all dense and BM25 views once. BM25 view runs are only internal for hybrid.
        dense_runs: dict[str, pd.DataFrame] = {}
        bm25_runs: dict[str, pd.DataFrame] = {}
        hybrid_runs: dict[str, pd.DataFrame] = {}

        for view_name, qdict in query_views.items():
            query_ids = list(qdict.keys())
            query_texts = [qdict[qid] for qid in query_ids]
            q_cache = CACHE_DIR / f"query_{safe_cache_name(spec.short_name + '_' + args.index_mode + '_' + view_name)}.npy"
            q_emb = encode_texts_cached(
                model=model,
                texts=query_texts,
                prefix=spec.query_prefix,
                cache_path=q_cache,
                batch_size=args.batch_size,
                device=args.device,
                force=args.force_recompute_embeddings,
                description=f"queries {view_name} for {spec.short_name}",
            )

            dense_method = f"dense_{spec.short_name}_{args.index_mode}_{view_name}"
            dense_run = dense_retrieve_from_embeddings(
                method=dense_method,
                query_ids=query_ids,
                query_emb=q_emb,
                doc_emb=doc_emb,
                doc_ids=doc_ids,
                top_k=args.top_k,
                query_batch_size=args.query_batch_size,
            )
            dense_runs[view_name] = dense_run
            save_run(dense_run, dense_method)

            bm25_method = f"bm25_internal_{args.index_mode}_{view_name}"
            bm25_run = bm25_retrieve(method=bm25_method, query_texts=qdict, bm25=bm25, doc_ids=doc_ids, top_k=args.top_k)
            bm25_runs[view_name] = bm25_run

            hybrid_method = f"hybrid_{spec.short_name}_{args.index_mode}_dw{dense_weight_tag}_{view_name}"
            hybrid_run = hybrid_fuse_scores(
                method=hybrid_method,
                bm25_run=bm25_run,
                dense_run=dense_run,
                dense_weight=args.dense_weight,
                top_k=args.top_k,
            )
            hybrid_runs[view_name] = hybrid_run
            save_run(hybrid_run, hybrid_method)

        # Dense direct replacement evaluation.
        dense_baseline = f"dense_{spec.short_name}_{args.index_mode}_original"
        for view_name in ["original", *replacement_views]:
            method = f"dense_{spec.short_name}_{args.index_mode}_{view_name}"
            all_per_query.append(
                evaluate_run_per_query(
                    dense_runs[view_name], qrels, query_task, method,
                    group_name=f"dense_{spec.short_name}_{args.index_mode}",
                    baseline_method=dense_baseline,
                    model_name=spec.short_name,
                    retriever_family="dense",
                    index_mode=args.index_mode,
                )
            )

        # Hybrid direct replacement evaluation.
        hybrid_baseline = f"hybrid_{spec.short_name}_{args.index_mode}_dw{dense_weight_tag}_original"
        for view_name in ["original", *replacement_views]:
            method = f"hybrid_{spec.short_name}_{args.index_mode}_dw{dense_weight_tag}_{view_name}"
            all_per_query.append(
                evaluate_run_per_query(
                    hybrid_runs[view_name], qrels, query_task, method,
                    group_name=f"hybrid_{spec.short_name}_{args.index_mode}",
                    baseline_method=hybrid_baseline,
                    model_name=spec.short_name,
                    retriever_family="hybrid",
                    index_mode=args.index_mode,
                )
            )

        # Hybrid original-preserving RRF fusion.
        fusion_specs = {
            "original_deepseek_keyword_gpt_legal": ["original", "deepseek_keyword", "gpt_legal"],
            "original_all_nonhyde": ["original", *all_nonhyde_views],
        }
        for fusion_name, views in fusion_specs.items():
            method = f"hybrid_rrf_{spec.short_name}_{args.index_mode}_dw{dense_weight_tag}_{fusion_name}"
            fused = rrf_fuse(method, [hybrid_runs[v] for v in views], rrf_k=args.rrf_k, top_k=args.top_k)
            save_run(fused, method)
            all_per_query.append(
                evaluate_run_per_query(
                    fused, qrels, query_task, method,
                    group_name=f"hybrid_{spec.short_name}_{args.index_mode}",
                    baseline_method=hybrid_baseline,
                    model_name=spec.short_name,
                    retriever_family="hybrid_rrf",
                    index_mode=args.index_mode,
                )
            )

        # Free memory between models.
        del model

    per_query_all = pd.concat(all_per_query, ignore_index=True)
    summary_df, gains_df = build_candidate_summary(per_query_all, stats_iterations=args.stats_iterations, seed=args.seed)
    by_task_df = build_by_task(per_query_all)

    summary_path = OUT_DIR / f"legalbench_dense_hybrid_summary_{output_suffix}.csv"
    gains_path = OUT_DIR / f"legalbench_dense_hybrid_per_query_{output_suffix}.csv"
    by_task_path = OUT_DIR / f"legalbench_dense_hybrid_by_task_{output_suffix}.csv"
    md_path = OUT_DIR / f"legalbench_dense_hybrid_summary_{output_suffix}.md"

    summary_df.to_csv(summary_path, index=False)
    gains_df.to_csv(gains_path, index=False)
    by_task_df.to_csv(by_task_path, index=False)
    summary_df.to_markdown(md_path, index=False)

    print("\n" + "=" * 90)
    print("DENSE + HYBRID SUMMARY")
    print("=" * 90)
    display_cols = [
        "method", "group", "retriever_family", "model", "Recall@10", "Delta Recall@10", "pHolm Recall@10",
        "Recall@100", "Delta Recall@100", "Harm Recall@100 %", "pHolm Recall@100",
        "MRR@10", "Delta MRR@10", "pHolm MRR@10",
        "nDCG@10", "Delta nDCG@10", "Harm nDCG@10 %", "pHolm nDCG@10",
        "Holm significant positive metrics", "Holm significant negative metrics",
    ]
    display_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[display_cols].to_string(index=False))

    print("\nSaved:")
    print(f"  {summary_path}")
    print(f"  {gains_path}")
    print(f"  {by_task_path}")
    print(f"  {md_path}")
    print(f"  Runs: {RUNS_DIR}")
    print(f"  Embedding cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
