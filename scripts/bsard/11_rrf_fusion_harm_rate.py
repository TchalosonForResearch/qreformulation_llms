"""Compute harm rates for one-view BSARD RRF runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RUNS_DIR = Path("runs/bsard")
TABLE_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

BASELINE_PER_QUERY_PATH = TABLE_DIR / "bm25_original_test_per_query.csv"

FUSION_RUNS = [
    ("deepseek", "legal_rewrite", RUNS_DIR / "rrf_bm25_original_deepseek_legal_rewrite_test.tsv"),
    ("deepseek", "keyword_expansion", RUNS_DIR / "rrf_bm25_original_deepseek_keyword_expansion_test.tsv"),
    ("deepseek", "hyde_style", RUNS_DIR / "rrf_bm25_original_deepseek_hyde_style_test.tsv"),
    ("gpt", "legal_rewrite", RUNS_DIR / "rrf_bm25_original_gpt_legal_rewrite_test.tsv"),
    ("gpt", "keyword_expansion", RUNS_DIR / "rrf_bm25_original_gpt_keyword_expansion_test.tsv"),
    ("gpt", "hyde_style", RUNS_DIR / "rrf_bm25_original_gpt_hyde_style_test.tsv"),
]

METRICS = ["Recall@10", "Recall@100", "MRR@10", "nDCG@10"]
EPS = 1e-12


def load_qrels(path: Path) -> pd.DataFrame:
    """
    Charge les qrels BSARD.

    Format :
        query_id    iter    doc_id    relevance
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )


def build_qrels_dict(qrels_df: pd.DataFrame) -> dict[str, set[str]]:
    """
    Convertit les qrels en dictionnaire :
        query_id -> ensemble des documents pertinents.
    """
    qrels = {}

    for qid, group in qrels_df.groupby("query_id"):
        qrels[str(qid)] = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )

    return qrels


def load_run(path: Path) -> pd.DataFrame:
    """
    Charge un run TSV produit par nos scripts.

    Colonnes attendues :
        query_id, doc_id, rank, score, method
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing run file: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype={"query_id": str, "doc_id": str},
    )

    required = {"query_id", "doc_id", "rank"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["rank"] = df["rank"].astype(int)

    return df


def compute_query_metrics(
    qid: str,
    ranked_docs: list[str],
    relevant_docs: set[str],
) -> dict:
    """
    Calcule Recall@10, Recall@100, MRR@10, nDCG@10
    pour une seule requête.
    """
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

    return {
        "query_id": qid,
        "num_relevant": n_relevant,
        "Recall@10": recall_10,
        "Recall@100": recall_100,
        "MRR@10": mrr_10,
        "nDCG@10": ndcg_10,
        "first_relevant_rank": first_relevant_rank,
    }


def evaluate_run_per_query(run_df: pd.DataFrame, qrels: dict[str, set[str]]) -> pd.DataFrame:
    """
    Évalue un run query par query.
    """
    grouped_run = run_df.groupby("query_id", sort=False)

    rows = []

    for qid, relevant_docs in qrels.items():
        if qid in grouped_run.groups:
            ranked_docs = (
                grouped_run.get_group(qid)
                .sort_values("rank")["doc_id"]
                .astype(str)
                .tolist()
            )
        else:
            ranked_docs = []

        rows.append(
            compute_query_metrics(
                qid=qid,
                ranked_docs=ranked_docs,
                relevant_docs=relevant_docs,
            )
        )

    return pd.DataFrame(rows)


def classify_gain(gain: float) -> str:
    if gain > EPS:
        return "improved"
    if gain < -EPS:
        return "harmed"
    return "neutral"


def format_signed(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    if not BASELINE_PER_QUERY_PATH.exists():
        raise FileNotFoundError(
            f"Missing baseline per-query file: {BASELINE_PER_QUERY_PATH}"
        )

    print("Loading qrels...")
    qrels = build_qrels_dict(load_qrels(QRELS_PATH))

    print("Loading baseline per-query metrics...")
    baseline = pd.read_csv(
        BASELINE_PER_QUERY_PATH,
        dtype={"query_id": str},
    )

    baseline = baseline[["query_id", *METRICS]].copy()
    baseline = baseline.rename(columns={m: f"{m}_baseline" for m in METRICS})

    summary_rows = []
    paper_rows = []
    all_gain_rows = []

    for generator, query_type, run_path in FUSION_RUNS:
        print("\n" + "=" * 80)
        print(f"Evaluating fusion: original + {generator}/{query_type}")
        print(f"Run file: {run_path}")

        run_df = load_run(run_path)
        per_query = evaluate_run_per_query(run_df, qrels)

        per_query["generator"] = generator
        per_query["query_type"] = query_type
        per_query["fusion"] = "original_plus_one_reformulation"

        # Sauvegarde individuelle par fusion
        individual_path = (
            TABLE_DIR
            / f"rrf_bm25_original_{generator}_{query_type}_test_per_query.csv"
        )
        per_query.to_csv(individual_path, index=False)

        candidate = per_query[["query_id", *METRICS]].copy()
        candidate = candidate.rename(columns={m: f"{m}_candidate" for m in METRICS})

        merged = baseline.merge(candidate, on="query_id", how="inner")

        if len(merged) != len(baseline):
            raise ValueError(
                f"Query mismatch for {generator}/{query_type}: "
                f"{len(merged)} merged vs {len(baseline)} baseline."
            )

        merged.insert(1, "generator", generator)
        merged.insert(2, "query_type", query_type)
        merged.insert(3, "fusion", "original_plus_one_reformulation")

        for metric in METRICS:
            base_col = f"{metric}_baseline"
            cand_col = f"{metric}_candidate"
            gain_col = f"{metric}_gain"
            status_col = f"{metric}_status"

            merged[gain_col] = merged[cand_col] - merged[base_col]
            merged[status_col] = merged[gain_col].apply(classify_gain)

            num_queries = len(merged)
            num_improved = int((merged[gain_col] > 0).sum())
            num_harmed = int((merged[gain_col] < 0).sum())
            num_neutral = int((merged[gain_col] == 0).sum())

            mean_baseline = float(merged[base_col].mean())
            mean_candidate = float(merged[cand_col].mean())
            mean_gain = float(merged[gain_col].mean())

            summary_rows.append(
                {
                    "generator": generator,
                    "query_type": query_type,
                    "metric": metric,
                    "mean_baseline": mean_baseline,
                    "mean_candidate": mean_candidate,
                    "mean_gain": mean_gain,
                    "num_queries": num_queries,
                    "num_improved": num_improved,
                    "num_harmed": num_harmed,
                    "num_neutral": num_neutral,
                    "improve_rate": num_improved / num_queries,
                    "harm_rate": num_harmed / num_queries,
                    "neutral_rate": num_neutral / num_queries,
                }
            )

            paper_rows.append(
                {
                    "generator": generator,
                    "query_type": query_type,
                    "metric": metric,
                    "baseline": round(mean_baseline, 6),
                    "fusion_score": round(mean_candidate, 6),
                    "delta": format_signed(mean_gain),
                    "harm_%": round((num_harmed / num_queries) * 100, 2),
                    "improve_%": round((num_improved / num_queries) * 100, 2),
                    "neutral_%": round((num_neutral / num_queries) * 100, 2),
                    "improved/harmed/neutral": (
                        f"{num_improved}/{num_harmed}/{num_neutral}"
                    ),
                }
            )

        all_gain_rows.append(merged)

    summary_df = pd.DataFrame(summary_rows)
    paper_df = pd.DataFrame(paper_rows)
    gains_df = pd.concat(all_gain_rows, ignore_index=True)

    query_type_order = {
        "legal_rewrite": 1,
        "keyword_expansion": 2,
        "hyde_style": 3,
    }

    metric_order = {
        "Recall@10": 1,
        "Recall@100": 2,
        "MRR@10": 3,
        "nDCG@10": 4,
    }

    generator_order = {
        "deepseek": 1,
        "gpt": 2,
    }

    paper_df["generator_order"] = paper_df["generator"].map(generator_order)
    paper_df["query_type_order"] = paper_df["query_type"].map(query_type_order)
    paper_df["metric_order"] = paper_df["metric"].map(metric_order)

    paper_df = paper_df.sort_values(
        ["generator_order", "query_type_order", "metric_order"]
    )

    paper_df = paper_df.drop(
        columns=["generator_order", "query_type_order", "metric_order"]
    )

    summary_path = TABLE_DIR / "rrf_original_reformulation_harm_rate.csv"
    paper_path = TABLE_DIR / "rrf_original_reformulation_summary_for_paper.csv"
    gains_path = TABLE_DIR / "rrf_original_reformulation_per_query_all.csv"

    summary_df.to_csv(summary_path, index=False)
    paper_df.to_csv(paper_path, index=False)
    gains_df.to_csv(gains_path, index=False)

    print("\n" + "=" * 80)
    print("Fusion harm-rate summary for paper:")
    print(paper_df.to_string(index=False))

    print(f"\nSaved harm-rate summary to: {summary_path}")
    print(f"Saved paper summary to: {paper_path}")
    print(f"Saved all per-query gains to: {gains_path}")


if __name__ == "__main__":
    main()