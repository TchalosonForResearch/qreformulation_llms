"""Fuse the original BSARD query with one reformulation using reciprocal rank fusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_RUN = RUNS_DIR / "bm25_original_test.tsv"

CANDIDATE_RUNS = [
    ("deepseek", "legal_rewrite", RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv"),
    ("deepseek", "keyword_expansion", RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv"),
    ("deepseek", "hyde_style", RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv"),
    ("gpt", "legal_rewrite", RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv"),
    ("gpt", "keyword_expansion", RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv"),
    ("gpt", "hyde_style", RUNS_DIR / "bm25_gpt_hyde_style_test.tsv"),
]

RRF_K = 60
TOP_K = 1000


def load_run(path: Path, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Charge un run TSV produit par nos scripts.

    Nettoyages appliqués :
    - conversion du rang en entier ;
    - limitation aux rangs <= top_k ;
    - suppression des doublons query_id/doc_id en gardant le meilleur rang.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"query_id": str, "doc_id": str},
    )

    required = {"query_id", "doc_id", "rank", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["rank"] = df["rank"].astype(int)
    df["score"] = df["score"].astype(float)

    # Important : seuls les top_k premiers résultats doivent contribuer à RRF.
    df = df[df["rank"] <= top_k].copy()

    # Sécurité : si un document apparaît deux fois pour une même requête,
    # on garde son meilleur rang.
    df = df.sort_values(["query_id", "doc_id", "rank"])
    df = df.drop_duplicates(["query_id", "doc_id"], keep="first")

    return df


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
        qrels[str(qid)] = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )

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

        for k in [10, 100]:
            retrieved = set(ranked_docs[:k])
            recall = (
                len(retrieved.intersection(relevant_docs)) / n_relevant
                if n_relevant > 0
                else 0.0
            )
            metrics[f"Recall@{k}"].append(recall)

        rr = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                rr = 1.0 / rank
                break
        metrics["MRR@10"].append(rr)

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_docs[:10], start=1):
            if doc_id in relevant_docs:
                dcg += 1.0 / np.log2(rank + 1)

        idcg = sum(
            1.0 / np.log2(rank + 1)
            for rank in range(1, min(n_relevant, 10) + 1)
        )

        metrics["nDCG@10"].append(dcg / idcg if idcg > 0 else 0.0)

    return {m: float(np.mean(v)) for m, v in metrics.items()}


def rrf_fuse_two_runs(
    run_a: pd.DataFrame,
    run_b: pd.DataFrame,
    run_name: str,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Fusionne deux runs query par query avec RRF.

    Tri déterministe :
    - score RRF décroissant ;
    - doc_id croissant en cas d'égalité.
    """
    all_query_ids = sorted(set(run_a["query_id"]) | set(run_b["query_id"]))

    grouped_a = run_a.groupby("query_id", sort=False)
    grouped_b = run_b.groupby("query_id", sort=False)

    fused_rows = []

    for qid in all_query_ids:
        scores = {}

        if qid in grouped_a.groups:
            rows_a = grouped_a.get_group(qid)
            for _, row in rows_a.iterrows():
                doc_id = str(row["doc_id"])
                rank = int(row["rank"])
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        if qid in grouped_b.groups:
            rows_b = grouped_b.get_group(qid)
            for _, row in rows_b.iterrows():
                doc_id = str(row["doc_id"])
                rank = int(row["rank"])
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

        # Tri déterministe : score décroissant, doc_id croissant.
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

        for rank, (doc_id, score) in enumerate(ranked, start=1):
            fused_rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": score,
                    "method": run_name,
                }
            )

    return pd.DataFrame(fused_rows)


def main() -> None:
    print("Loading qrels...")
    qrels = build_qrels_dict(load_qrels(QRELS_PATH))

    print("Loading original BM25 run...")
    original_run = load_run(ORIGINAL_RUN, top_k=TOP_K)

    baseline_metrics = evaluate_run(original_run, qrels)

    all_metrics = [
        {
            "method": "bm25_original_test",
            "generator": "none",
            "query_type": "original",
            "fusion": "none",
            "rrf_k": None,
            **baseline_metrics,
        }
    ]

    print("\nBaseline BM25 original:")
    print(pd.DataFrame([all_metrics[0]]).to_string(index=False))

    for generator, query_type, candidate_path in CANDIDATE_RUNS:
        print("\n" + "=" * 80)
        print(f"Fusion: original + {generator}/{query_type}")

        candidate_run = load_run(candidate_path, top_k=TOP_K)

        run_name = f"rrf_bm25_original_{generator}_{query_type}_test"

        fused = rrf_fuse_two_runs(
            run_a=original_run,
            run_b=candidate_run,
            run_name=run_name,
            rrf_k=RRF_K,
            top_k=TOP_K,
        )

        output_run_path = RUNS_DIR / f"{run_name}.tsv"
        fused.to_csv(output_run_path, sep="\t", index=False)

        print(f"Saved fused run to: {output_run_path}")

        metrics = evaluate_run(fused, qrels)

        row = {
            "method": run_name,
            "generator": generator,
            "query_type": query_type,
            "fusion": "original_plus_one_reformulation",
            "rrf_k": RRF_K,
            **metrics,
        }

        all_metrics.append(row)

        print(pd.DataFrame([row]).to_string(index=False))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = OUT_DIR / "rrf_original_reformulation_test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print("\n" + "=" * 80)
    print(f"Saved metrics to: {metrics_path}")
    print("\nAll fusion metrics:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()