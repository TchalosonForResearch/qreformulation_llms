"""Run BM25 on each normalized BSARD reformulation view."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm


RAW_DIR = Path("data/raw/bsard")
REFORM_DIR = Path("data/processed/bsard/reformulations/normalized")
RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


GENERATORS = ["deepseek", "gpt"]
REFORMULATION_FIELDS = ["legal_rewrite", "keyword_expansion", "hyde_style"]


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


def simple_tokenize(text: str) -> list[str]:
    """
    Tokenisation simple pour BM25.

    On garde volontairement la même tokenisation que pour la baseline
    BM25 originale. C'est très important : si la tokenisation change,
    on ne saura plus si les différences viennent de la reformulation
    ou du prétraitement.
    """
    text = text.lower()
    return re.findall(r"\b\w+\b", text, flags=re.UNICODE)


def load_qrels(path: Path) -> pd.DataFrame:
    """
    Charge les qrels BSARD.

    Format TREC :
        query_id    iter    doc_id    relevance

    Les fichiers qrels n'ont pas d'en-tête.
    """
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


def evaluate_run(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> dict[str, float]:
    """
    Calcule les métriques globales :
    Recall@10, Recall@100, MRR@10, nDCG@10.
    """
    metrics = {
        "Recall@10": [],
        "Recall@100": [],
        "MRR@10": [],
        "nDCG@10": [],
    }

    grouped = run_df.groupby("query_id", sort=False)

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

        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        ndcg = dcg / idcg if idcg > 0 else 0.0

        metrics["Recall@10"].append(recall_10)
        metrics["Recall@100"].append(recall_100)
        metrics["MRR@10"].append(rr)
        metrics["nDCG@10"].append(ndcg)

    return {name: float(np.mean(values)) for name, values in metrics.items()}


def build_bm25_index(corpus: list[dict]) -> tuple[BM25Okapi, list[str]]:
    """
    Construit l'index BM25 une seule fois.

    Le corpus ne change pas entre les reformulations.
    Ce qui change, c'est seulement le texte de requête.
    """
    doc_ids = [str(doc["doc_id"]) for doc in corpus]

    # Même choix que la baseline originale :
    # le champ 'text' contient référence/titre + contenu de l'article.
    doc_texts = [doc.get("text", "") or "" for doc in corpus]

    print("Tokenizing corpus...")
    tokenized_corpus = [simple_tokenize(text) for text in tqdm(doc_texts)]

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, doc_ids


def retrieve_for_reformulation_field(
    bm25: BM25Okapi,
    doc_ids: list[str],
    reformulations: list[dict],
    generator: str,
    field: str,
    top_k: int,
) -> pd.DataFrame:
    """
    Lance BM25 avec un champ de reformulation donné.

    Exemple :
        generator = "gpt"
        field = "keyword_expansion"

    produit :
        bm25_gpt_keyword_expansion_test.tsv
    """
    run_rows = []

    for row in tqdm(reformulations, desc=f"{generator}/{field}"):
        qid = str(row["query_id"])
        query_text = row.get(field, "") or ""

        tokenized_query = simple_tokenize(query_text)

        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        for rank, idx in enumerate(top_indices, start=1):
            run_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_ids[idx],
                    "rank": rank,
                    "score": float(scores[idx]),
                    "method": f"bm25_{generator}_{field}",
                    "generator": generator,
                    "query_type": field,
                }
            )

    return pd.DataFrame(run_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BM25 on existing BSARD reformulations."
    )

    parser.add_argument(
        "--generator",
        choices=["deepseek", "gpt", "all"],
        default="all",
        help="Which generator to run.",
    )

    parser.add_argument(
        "--field",
        choices=["legal_rewrite", "keyword_expansion", "hyde_style", "all"],
        default="all",
        help="Which reformulation field to run.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1000,
        help="Number of documents retrieved per query.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    selected_generators = (
        GENERATORS if args.generator == "all" else [args.generator]
    )

    selected_fields = (
        REFORMULATION_FIELDS if args.field == "all" else [args.field]
    )

    corpus_path = RAW_DIR / "corpus.jsonl"
    qrels_path = RAW_DIR / "qrels_test.tsv"

    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_path}")

    if not qrels_path.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_path}")

    print("Loading corpus...")
    corpus = read_jsonl(corpus_path)

    print("Loading qrels...")
    qrels = build_qrels_dict(load_qrels(qrels_path))

    print(f"Corpus documents: {len(corpus)}")
    print(f"Qrels queries: {len(qrels)}")

    bm25, doc_ids = build_bm25_index(corpus)

    all_metrics_rows = []

    for generator in selected_generators:
        reform_path = REFORM_DIR / f"{generator}_test.jsonl"

        if not reform_path.exists():
            raise FileNotFoundError(f"Missing reformulation file: {reform_path}")

        print("\n" + "=" * 80)
        print(f"Loading reformulations for generator: {generator}")
        print(f"File: {reform_path}")

        reformulations = read_jsonl(reform_path)
        print(f"Reformulations: {len(reformulations)}")

        for field in selected_fields:
            print("\n" + "-" * 80)
            print(f"Running BM25 for generator={generator}, field={field}")

            run_df = retrieve_for_reformulation_field(
                bm25=bm25,
                doc_ids=doc_ids,
                reformulations=reformulations,
                generator=generator,
                field=field,
                top_k=args.top_k,
            )

            run_name = f"bm25_{generator}_{field}_test"

            run_path = RUNS_DIR / f"{run_name}.tsv"
            run_df.to_csv(run_path, sep="\t", index=False)

            print(f"Saved run to: {run_path}")

            metrics = evaluate_run(run_df, qrels)

            metrics_row = {
                "method": f"bm25_{generator}_{field}",
                "generator": generator,
                "query_type": field,
                **metrics,
            }

            all_metrics_rows.append(metrics_row)

            print("Metrics:")
            print(pd.DataFrame([metrics_row]).to_string(index=False))

    metrics_df = pd.DataFrame(all_metrics_rows)

    summary_path = OUT_DIR / "bm25_reformulations_test_metrics.csv"

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, metrics_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["method", "generator", "query_type"],
            keep="last",
        )
        combined.to_csv(summary_path, index=False)
    else:
        metrics_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved metrics summary to: {summary_path}")

    print("\nAll metrics from this run:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()