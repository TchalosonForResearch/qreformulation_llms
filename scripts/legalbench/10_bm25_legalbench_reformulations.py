"""Run BM25 on the normalized LegalBench reformulation views."""

from __future__ import annotations

import argparse
import gc
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


DATA_DIR = Path("data/processed/legalbench/rag_mini")
REFORMULATIONS_PATH = (
    DATA_DIR / "reformulations/normalized/all_generators_mini.jsonl"
)

RUNS_DIR = Path("runs/legalbench")
OUT_DIR = Path("outputs/tables/legalbench")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENERATORS = ["deepseek", "gpt"]
QUERY_TYPES = ["legal_rewrite", "keyword_expansion", "hyde_style"]


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


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenisation simple pour BM25.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def clean_metadata_text(text: str) -> str:
    """
    Transforme un chemin de fichier en texte indexable.

    Exemple :
      privacy_qa/Fiverr.txt

    devient :
      privacy qa Fiverr
    """
    text = text.replace("\\", "/")
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace(".txt", " ")
    text = text.replace(".pdf", " ")
    text = text.replace("||", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def build_index_text(doc: dict) -> str:
    """
    Variante canonique LegalBench :
      filepath_plus_chunk
    """
    file_path = str(doc.get("file_path", "") or "")
    chunk_text = str(doc.get("text", "") or "")

    filepath_text = clean_metadata_text(file_path)

    return f"{filepath_text}\n{chunk_text}"


def load_qrels(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )


def build_qrels_dict(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    qrels = {}

    for qid, group in qrels_df.groupby("query_id"):
        relevant_docs = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )
        qrels[str(qid)] = relevant_docs

    return qrels


def evaluate_run_per_query(
    run_df: pd.DataFrame,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
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

        retrieved_10 = set(ranked_docs[:10])
        retrieved_100 = set(ranked_docs[:100])

        recall_10 = (
            len(retrieved_10.intersection(relevant_docs)) / n_relevant
            if n_relevant > 0
            else 0.0
        )

        recall_100 = (
            len(retrieved_100.intersection(relevant_docs)) / n_relevant
            if n_relevant > 0
            else 0.0
        )

        mrr_10 = 0.0
        first_relevant_rank = None

        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                mrr_10 = 1.0 / rank
                first_relevant_rank = rank
                break

        dcg_10 = 0.0

        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg_10 += 1.0 / np.log2(rank + 1)

        idcg_10 = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        ndcg_10 = dcg_10 / idcg_10 if idcg_10 > 0 else 0.0

        rows.append(
            {
                "query_id": qid,
                "task": query_task.get(qid, "unknown"),
                "num_relevant": n_relevant,
                "Recall@10": recall_10,
                "Recall@100": recall_100,
                "MRR@10": mrr_10,
                "nDCG@10": ndcg_10,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    return pd.DataFrame(rows)


def summarize_global(per_query_df: pd.DataFrame) -> dict:
    return {
        "num_queries": int(len(per_query_df)),
        "Recall@10": float(per_query_df["Recall@10"].mean()),
        "Recall@100": float(per_query_df["Recall@100"].mean()),
        "MRR@10": float(per_query_df["MRR@10"].mean()),
        "nDCG@10": float(per_query_df["nDCG@10"].mean()),
    }


def summarize_by_task(per_query_df: pd.DataFrame) -> pd.DataFrame:
    return (
        per_query_df
        .groupby("task")
        .agg(
            num_queries=("query_id", "count"),
            mean_num_relevant=("num_relevant", "mean"),
            **{
                "Recall@10": ("Recall@10", "mean"),
                "Recall@100": ("Recall@100", "mean"),
                "MRR@10": ("MRR@10", "mean"),
                "nDCG@10": ("nDCG@10", "mean"),
            },
        )
        .reset_index()
    )


def make_reformulation_map(rows: list[dict]) -> dict[tuple[str, str], dict[str, dict]]:
    """
    Retourne :
      (generator, query_type) -> query_id -> row

    Exemple :
      ("deepseek", "legal_rewrite") -> {"contractnli_00031": {...}}
    """
    mapping = {}

    for generator in GENERATORS:
        for query_type in QUERY_TYPES:
            mapping[(generator, query_type)] = {}

    for row in rows:
        if row.get("validation_status") != "valid":
            continue

        generator = str(row.get("generator", ""))

        if generator not in GENERATORS:
            continue

        query_id = str(row["query_id"])

        for query_type in QUERY_TYPES:
            text = row.get(query_type, "")

            if isinstance(text, str) and text.strip():
                mapping[(generator, query_type)][query_id] = row

    return mapping


def run_one_reformulation_setting(
    *,
    generator: str,
    query_type: str,
    reformulation_rows_by_qid: dict[str, dict],
    queries: list[dict],
    doc_ids: list[str],
    bm25: BM25Okapi,
    qrels: dict[str, set[str]],
    query_task: dict[str, str],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method = f"bm25_{generator}_{query_type}"

    print("\n" + "=" * 80)
    print(f"Running {method}")
    print("=" * 80)

    missing_query_text = []

    run_rows = []
    effective_top_k = min(top_k, len(doc_ids))

    for query in tqdm(queries, desc=method):
        qid = str(query["query_id"])

        row = reformulation_rows_by_qid.get(qid)

        if row is None:
            missing_query_text.append(qid)
            query_text = ""
        else:
            query_text = row.get(query_type, "") or ""

        tokenized_query = simple_tokenize(query_text)

        scores = bm25.get_scores(tokenized_query)

        if effective_top_k < len(scores):
            candidate_indices = np.argpartition(-scores, effective_top_k - 1)[:effective_top_k]
        else:
            candidate_indices = np.arange(len(scores))

        ranked_indices = sorted(
            candidate_indices,
            key=lambda idx: (-float(scores[idx]), doc_ids[idx]),
        )

        for rank, idx in enumerate(ranked_indices, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                    "method": method,
                    "generator": generator,
                    "query_type": query_type,
                }
            )

    if missing_query_text:
        print(
            f"WARNING: missing reformulation text for "
            f"{len(missing_query_text)} queries."
        )
        print(f"First missing query_ids: {missing_query_text[:10]}")

    run_df = pd.DataFrame(run_rows)

    run_path = RUNS_DIR / f"{method}_mini.tsv"
    run_df.to_csv(run_path, sep="\t", index=False)

    print(f"Saved run to: {run_path}")

    per_query_df = evaluate_run_per_query(
        run_df=run_df,
        qrels=qrels,
        query_task=query_task,
    )

    per_query_df.insert(0, "method", method)
    per_query_df.insert(1, "generator", generator)
    per_query_df.insert(2, "query_type", query_type)

    global_metrics = {
        "method": method,
        "dataset": "legalbench_rag_mini",
        "generator": generator,
        "query_type": query_type,
        **summarize_global(per_query_df),
    }

    global_df = pd.DataFrame([global_metrics])

    by_task_df = summarize_by_task(per_query_df)
    by_task_df.insert(0, "method", method)
    by_task_df.insert(1, "generator", generator)
    by_task_df.insert(2, "query_type", query_type)

    per_query_path = OUT_DIR / f"{method}_mini_per_query.csv"
    metrics_path = OUT_DIR / f"{method}_mini_metrics.csv"
    by_task_path = OUT_DIR / f"{method}_mini_by_task.csv"

    per_query_df.to_csv(per_query_path, index=False)
    global_df.to_csv(metrics_path, index=False)
    by_task_df.to_csv(by_task_path, index=False)

    print("\nGlobal metrics:")
    print(global_df.to_string(index=False))

    print("\nMetrics by task:")
    print(by_task_df.to_string(index=False))

    return global_df, by_task_df, per_query_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BM25 with LegalBench-RAG mini reformulations."
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of retrieved chunks saved per query.",
    )

    parser.add_argument(
        "--generators",
        nargs="*",
        default=GENERATORS,
        choices=GENERATORS,
        help="Generators to evaluate.",
    )

    parser.add_argument(
        "--query-types",
        nargs="*",
        default=QUERY_TYPES,
        choices=QUERY_TYPES,
        help="Reformulation fields to evaluate.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    corpus_path = DATA_DIR / "corpus.jsonl"
    queries_path = DATA_DIR / "queries.jsonl"
    qrels_path = DATA_DIR / "qrels.tsv"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    if not REFORMULATIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing reformulations file: {REFORMULATIONS_PATH}\n"
            "Run script 09_inspect_normalize_legalbench_reformulations.py first."
        )

    print("=" * 80)
    print("BM25 LegalBench-RAG mini reformulations")
    print("=" * 80)

    print(f"Corpus: {corpus_path}")
    print(f"Queries: {queries_path}")
    print(f"Qrels: {qrels_path}")
    print(f"Reformulations: {REFORMULATIONS_PATH}")
    print(f"Top-k: {args.top_k}")
    print(f"Generators: {args.generators}")
    print(f"Query types: {args.query_types}")
    print("Index text variant: filepath_plus_chunk")

    corpus = read_jsonl(corpus_path)
    queries = read_jsonl(queries_path)
    qrels_df = load_qrels(qrels_path)
    qrels = build_qrels_dict(qrels_df)
    reformulation_rows = read_jsonl(REFORMULATIONS_PATH)

    query_task = {
        str(query["query_id"]): str(query.get("task", "unknown"))
        for query in queries
    }

    doc_ids = [str(doc["doc_id"]) for doc in corpus]

    print("\nDataset sizes:")
    print(f"Corpus chunks: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels queries: {len(qrels)}")
    print(f"Qrels rows: {len(qrels_df)}")
    print(f"Reformulation rows: {len(reformulation_rows)}")

    reformulation_map = make_reformulation_map(reformulation_rows)

    print("\nReformulation coverage by setting:")
    for generator in args.generators:
        for query_type in args.query_types:
            count = len(reformulation_map[(generator, query_type)])
            print(f"  {generator}/{query_type}: {count}")

    print("\nBuilding canonical LegalBench BM25 index: filepath_plus_chunk")
    index_texts = [
        build_index_text(doc)
        for doc in tqdm(corpus, desc="Index texts")
    ]

    print("Tokenizing corpus...")
    tokenized_corpus = [
        simple_tokenize(text)
        for text in tqdm(index_texts, desc="Tokenizing")
    ]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    all_global = []
    all_by_task = []
    all_per_query = []

    for generator in args.generators:
        for query_type in args.query_types:
            global_df, by_task_df, per_query_df = run_one_reformulation_setting(
                generator=generator,
                query_type=query_type,
                reformulation_rows_by_qid=reformulation_map[(generator, query_type)],
                queries=queries,
                doc_ids=doc_ids,
                bm25=bm25,
                qrels=qrels,
                query_task=query_task,
                top_k=args.top_k,
            )

            all_global.append(global_df)
            all_by_task.append(by_task_df)
            all_per_query.append(per_query_df)

    metrics_df = pd.concat(all_global, ignore_index=True)
    by_task_all_df = pd.concat(all_by_task, ignore_index=True)
    per_query_all_df = pd.concat(all_per_query, ignore_index=True)

    metrics_path = OUT_DIR / "bm25_reformulations_mini_metrics.csv"
    by_task_path = OUT_DIR / "bm25_reformulations_mini_by_task.csv"
    per_query_all_path = OUT_DIR / "bm25_reformulations_mini_per_query_all.csv"

    metrics_df.to_csv(metrics_path, index=False)
    by_task_all_df.to_csv(by_task_path, index=False)
    per_query_all_df.to_csv(per_query_all_path, index=False)

    print("\n" + "=" * 80)
    print("All BM25 reformulation metrics:")
    print(metrics_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("All BM25 reformulation metrics by task:")
    print(by_task_all_df.to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(by_task_path)
    print(per_query_all_path)

    print("\nNext:")
    print("  scripts/legalbench/11_diagnose_anchor_preservation.py")

    del index_texts
    del tokenized_corpus
    del bm25
    gc.collect()


if __name__ == "__main__":
    main()