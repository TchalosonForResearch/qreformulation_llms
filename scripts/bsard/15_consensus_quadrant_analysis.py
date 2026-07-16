"""Analyse retrieval consensus patterns across BSARD reformulation views."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUNS_DIR = Path("runs/bsard")
TABLE_DIR = Path("outputs/tables/bsard")
QRELS_PATH = Path("data/raw/bsard/qrels_test.tsv")

TABLE_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_RUN_PATH = RUNS_DIR / "bm25_original_test.tsv"

REFORMULATION_RUNS = {
    "deepseek_legal_rewrite": RUNS_DIR / "bm25_deepseek_legal_rewrite_test.tsv",
    "deepseek_keyword_expansion": RUNS_DIR / "bm25_deepseek_keyword_expansion_test.tsv",
    "deepseek_hyde_style": RUNS_DIR / "bm25_deepseek_hyde_style_test.tsv",
    "gpt_legal_rewrite": RUNS_DIR / "bm25_gpt_legal_rewrite_test.tsv",
    "gpt_keyword_expansion": RUNS_DIR / "bm25_gpt_keyword_expansion_test.tsv",
    "gpt_hyde_style": RUNS_DIR / "bm25_gpt_hyde_style_test.tsv",
}

GROUPS = {
    "deepseek_only": [
        "deepseek_legal_rewrite",
        "deepseek_keyword_expansion",
        "deepseek_hyde_style",
    ],
    "gpt_only": [
        "gpt_legal_rewrite",
        "gpt_keyword_expansion",
        "gpt_hyde_style",
    ],
    "all_generators": [
        "deepseek_legal_rewrite",
        "deepseek_keyword_expansion",
        "deepseek_hyde_style",
        "gpt_legal_rewrite",
        "gpt_keyword_expansion",
        "gpt_hyde_style",
    ],
}


def load_run(path: Path, cutoff: int) -> pd.DataFrame:
    """
    Charge un run TSV et ne garde que le top cutoff.

    On supprime aussi les doublons query_id/doc_id par sécurité.
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
    df = df[df["rank"] <= cutoff].copy()

    df = df.sort_values(["query_id", "doc_id", "rank"])
    df = df.drop_duplicates(["query_id", "doc_id"], keep="first")

    return df


def load_qrels(path: Path) -> dict[str, set[str]]:
    """
    Charge les qrels BSARD :
        query_id    iter    doc_id    relevance
    """
    qrels_df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["query_id", "iter", "doc_id", "relevance"],
        dtype={"query_id": str, "doc_id": str, "relevance": int},
    )

    qrels = {}

    for qid, group in qrels_df.groupby("query_id"):
        qrels[str(qid)] = set(
            group.loc[group["relevance"] > 0, "doc_id"].astype(str)
        )

    return qrels


def run_to_rank_dict(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Transforme un run en dictionnaire :
        query_id -> {doc_id -> rank}
    """
    result = {}

    for qid, group in df.groupby("query_id"):
        result[str(qid)] = {
            str(row["doc_id"]): int(row["rank"])
            for _, row in group.iterrows()
        }

    return result


def classify_quadrant(anchor_high: bool, consensus_high: bool) -> str:
    if anchor_high and consensus_high:
        return "anchor_high__consensus_high"
    if anchor_high and not consensus_high:
        return "anchor_high__consensus_low"
    if not anchor_high and consensus_high:
        return "anchor_low__consensus_high"
    return "anchor_low__consensus_low"


def make_markdown_table(df: pd.DataFrame) -> str:
    """
    Produit un tableau Markdown sans dépendre de tabulate.
    """
    columns = list(df.columns)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"

    rows = []

    for _, row in df.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator, *rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze anchor/consensus relevance quadrants for BSARD."
    )

    parser.add_argument(
        "--cutoff",
        type=int,
        default=100,
        help="Top-K cutoff used to define anchor_high and reformulation votes.",
    )

    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of reformulation votes for consensus_high.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cutoff = args.cutoff
    min_votes = args.min_votes

    print(f"Consensus analysis with cutoff={cutoff}, min_votes={min_votes}")

    print("Loading qrels...")
    qrels = load_qrels(QRELS_PATH)

    print("Loading original run...")
    original_df = load_run(ORIGINAL_RUN_PATH, cutoff=cutoff)
    original_ranks = run_to_rank_dict(original_df)

    print("Loading reformulation runs...")
    reform_rank_dicts = {}

    for name, path in REFORMULATION_RUNS.items():
        df = load_run(path, cutoff=cutoff)
        reform_rank_dicts[name] = run_to_rank_dict(df)
        print(f"Loaded {name}: {len(df)} rows")

    per_doc_rows = []

    all_query_ids = sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else x)

    for group_name, reform_names in GROUPS.items():
        print("\n" + "=" * 80)
        print(f"Analyzing group: {group_name}")
        print(f"Reformulation views: {reform_names}")

        for qid in all_query_ids:
            relevant_docs = qrels.get(qid, set())

            candidate_docs = set()

            # Documents de l'ancre
            anchor_docs_for_q = set(original_ranks.get(qid, {}).keys())
            candidate_docs.update(anchor_docs_for_q)

            # Documents des reformulations
            for reform_name in reform_names:
                docs_for_q = set(reform_rank_dicts[reform_name].get(qid, {}).keys())
                candidate_docs.update(docs_for_q)

            for doc_id in candidate_docs:
                anchor_rank = original_ranks.get(qid, {}).get(doc_id)
                anchor_high = anchor_rank is not None

                votes = 0
                vote_sources = []

                for reform_name in reform_names:
                    rank = reform_rank_dicts[reform_name].get(qid, {}).get(doc_id)

                    if rank is not None:
                        votes += 1
                        vote_sources.append(reform_name)

                consensus_high = votes >= min_votes
                quadrant = classify_quadrant(anchor_high, consensus_high)

                per_doc_rows.append(
                    {
                        "group": group_name,
                        "query_id": qid,
                        "doc_id": doc_id,
                        "is_relevant": doc_id in relevant_docs,
                        "anchor_rank": anchor_rank,
                        "anchor_high": anchor_high,
                        "consensus_votes": votes,
                        "consensus_high": consensus_high,
                        "quadrant": quadrant,
                        "vote_sources": ",".join(vote_sources),
                    }
                )

    per_doc_df = pd.DataFrame(per_doc_rows)

    summary_rows = []

    for (group_name, quadrant), group in per_doc_df.groupby(["group", "quadrant"]):
        num_pairs = len(group)
        num_relevant = int(group["is_relevant"].sum())
        relevance_rate = num_relevant / num_pairs if num_pairs > 0 else 0.0
        num_queries = group["query_id"].nunique()

        summary_rows.append(
            {
                "group": group_name,
                "quadrant": quadrant,
                "num_query_doc_pairs": num_pairs,
                "num_queries": num_queries,
                "num_relevant": num_relevant,
                "relevance_rate": relevance_rate,
                "relevance_%": round(relevance_rate * 100, 4),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    # Enrichment par rapport au quadrant le plus faible du même groupe.
    enrichment_values = []

    for group_name, group in summary_df.groupby("group"):
        low_low = group[group["quadrant"] == "anchor_low__consensus_low"]

        if len(low_low) == 0:
            base_rate = None
        else:
            base_rate = float(low_low.iloc[0]["relevance_rate"])

        for _, row in group.iterrows():
            if base_rate is None or base_rate == 0:
                enrichment = None
            else:
                enrichment = float(row["relevance_rate"]) / base_rate

            enrichment_values.append(
                {
                    "group": row["group"],
                    "quadrant": row["quadrant"],
                    "enrichment_vs_low_low": enrichment,
                }
            )

    enrichment_df = pd.DataFrame(enrichment_values)

    summary_df = summary_df.merge(
        enrichment_df,
        on=["group", "quadrant"],
        how="left",
    )

    summary_df["enrichment_vs_low_low"] = summary_df["enrichment_vs_low_low"].apply(
        lambda x: None if pd.isna(x) else round(float(x), 3)
    )

    quadrant_order = {
        "anchor_high__consensus_high": 1,
        "anchor_high__consensus_low": 2,
        "anchor_low__consensus_high": 3,
        "anchor_low__consensus_low": 4,
    }

    group_order = {
        "deepseek_only": 1,
        "gpt_only": 2,
        "all_generators": 3,
    }

    summary_df["group_order"] = summary_df["group"].map(group_order)
    summary_df["quadrant_order"] = summary_df["quadrant"].map(quadrant_order)

    summary_df = summary_df.sort_values(
        ["group_order", "quadrant_order"]
    ).drop(columns=["group_order", "quadrant_order"])

    per_doc_path = TABLE_DIR / "consensus_quadrant_per_doc.csv"
    summary_path = TABLE_DIR / "consensus_quadrant_summary.csv"
    summary_md_path = TABLE_DIR / "consensus_quadrant_summary.md"

    per_doc_df.to_csv(per_doc_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    markdown_text = make_markdown_table(summary_df)
    summary_md_path.write_text(markdown_text, encoding="utf-8")

    print("\n" + "=" * 80)
    print("Consensus quadrant summary:")
    print(summary_df.to_string(index=False))

    print("\nSaved files:")
    print(per_doc_path)
    print(summary_path)
    print(summary_md_path)


if __name__ == "__main__":
    main()