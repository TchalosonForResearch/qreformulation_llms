"""Build multi-view BSARD RRF runs that retain the original query."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
OUT_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

OUT_DIR.mkdir(parents=True, exist_ok=True)

RRF_K = 60
TOP_K = 1000

ORIGINAL_RUN = RUNS_DIR / "bm25_original_test.tsv"

FUSION_GROUPS = [
    {
        "name": "rrf_bm25_original_deepseek_all_reformulations_test",
        "generator": "deepseek",
        "fusion_type": "original_plus_all_deepseek",
        "run_paths": [
            ORIGINAL_RUN,
            RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
        ],
    },
    {
        "name": "rrf_bm25_original_gpt_all_reformulations_test",
        "generator": "gpt",
        "fusion_type": "original_plus_all_gpt",
        "run_paths": [
            ORIGINAL_RUN,
            RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
        ],
    },
    {
        "name": "rrf_bm25_original_all_generators_all_reformulations_test",
        "generator": "deepseek+gpt",
        "fusion_type": "original_plus_all_generators",
        "run_paths": [
            ORIGINAL_RUN,
            RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
            RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
            RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
            RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
        ],
    },
]


def load_run(path: Path, top_k: int = TOP_K) -> pd.DataFrame:
    """
    Charge un run TSV.

    Nettoyages :
    - garde seulement les documents jusqu'au rang top_k ;
    - supprime les doublons query_id/doc_id ;
    - garde le meilleur rang si doublon.
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

    df = df[df["rank"] <= top_k].copy()

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
    Calcule Recall@10, Recall@100, MRR@10, nDCG@10.
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


def rrf_fuse_runs(
    runs: list[pd.DataFrame],
    run_name: str,
    rrf_k: int = RRF_K,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """
    Fusionne plusieurs runs par RRF.

    Tri déterministe :
    - score RRF décroissant ;
    - doc_id croissant en cas d'égalité.
    """
    all_query_ids = sorted(
        set().union(*[set(run["query_id"]) for run in runs])
    )

    grouped_runs = [
        run.groupby("query_id", sort=False)
        for run in runs
    ]

    fused_rows = []

    for qid in all_query_ids:
        scores = {}

        for grouped in grouped_runs:
            if qid not in grouped.groups:
                continue

            rows = grouped.get_group(qid)

            for _, row in rows.iterrows():
                doc_id = str(row["doc_id"])
                rank = int(row["rank"])

                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)

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

    print("Loading original baseline...")
    original_run = load_run(ORIGINAL_RUN, top_k=TOP_K)
    baseline_metrics = evaluate_run(original_run, qrels)

    metrics_rows = [
        {
            "method": "bm25_original_test",
            "generator": "none",
            "fusion_type": "none",
            "num_runs_fused": 1,
            "rrf_k": None,
            **baseline_metrics,
        }
    ]

    print("\nBaseline:")
    print(pd.DataFrame([metrics_rows[0]]).to_string(index=False))

    for group in FUSION_GROUPS:
        print("\n" + "=" * 80)
        print(f"Fusion group: {group['name']}")
        print(f"Runs fused: {len(group['run_paths'])}")

        runs = [load_run(path, top_k=TOP_K) for path in group["run_paths"]]

        fused = rrf_fuse_runs(
            runs=runs,
            run_name=group["name"],
            rrf_k=RRF_K,
            top_k=TOP_K,
        )

        output_run_path = RUNS_DIR / f"{group['name']}.tsv"
        fused.to_csv(output_run_path, sep="\t", index=False)

        metrics = evaluate_run(fused, qrels)

        row = {
            "method": group["name"],
            "generator": group["generator"],
            "fusion_type": group["fusion_type"],
            "num_runs_fused": len(group["run_paths"]),
            "rrf_k": RRF_K,
            **metrics,
        }

        metrics_rows.append(row)

        print(f"Saved fused run to: {output_run_path}")
        print(pd.DataFrame([row]).to_string(index=False))

    metrics_df = pd.DataFrame(metrics_rows)

    output_metrics_path = OUT_DIR / "rrf_original_all_reformulations_test_metrics.csv"
    metrics_df.to_csv(output_metrics_path, index=False)

    print("\n" + "=" * 80)
    print("All multi-reformulation RRF metrics:")
    print(metrics_df.to_string(index=False))

    print(f"\nSaved metrics to: {output_metrics_path}")


if __name__ == "__main__":
    main()