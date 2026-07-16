"""Prepare the LegalBench-RAG mini corpus, queries, and qrels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


RAW_DIR = Path("data/raw/legalbench")
CORPUS_DIR = RAW_DIR / "corpus"
BENCHMARKS_DIR = RAW_DIR / "benchmarks"

OUT_DIR = Path("data/processed/legalbench/rag_mini")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "contractnli",
    "cuad",
    "maud",
    "privacy_qa",
]

QRELS_DETAIL_FIELDS = [
    "query_id",
    "task",
    "snippet_index",
    "raw_file_path",
    "resolved_file_path",
    "file_path",
    "span_start",
    "span_end",
    "answer_length",
    "required_overlap_chars",
    "doc_id",
    "chunk_start",
    "chunk_end",
    "overlap_chars",
    "overlap_fraction_of_answer",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")


def get_json_items(data: Any) -> list:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        candidate_keys = [
            "tests",
            "test_cases",
            "examples",
            "data",
            "queries",
            "items",
        ]

        for key in candidate_keys:
            if key in data and isinstance(data[key], list):
                return data[key]

        for value in data.values():
            if isinstance(value, list):
                return value

    return []


def make_doc_id(file_path: str, start: int, end: int) -> str:
    """
    Crée un identifiant stable pour un chunk.

    On utilise SHA256 complet du chemin de fichier normalisé.
    C'est plus sûr qu'un hash court tronqué.
    """
    normalized_path = file_path.replace("\\", "/")
    file_hash = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()
    return f"lb_{file_hash}_{start}_{end}"


def chunk_document(
    *,
    task: str,
    file_path: str,
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    rows = []

    start = 0
    chunk_index = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]

        doc_id = make_doc_id(file_path, start, end)

        rows.append(
            {
                "doc_id": doc_id,
                "task": task,
                "file_path": file_path,
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
            }
        )

        if end >= text_length:
            break

        start = end - overlap
        chunk_index += 1

    return rows


def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def normalize_path_key(path: str) -> str:
    """
    Normalise un chemin pour faire du matching souple.

    On garde uniquement les caractères alphanumériques en minuscules.
    Cela permet de matcher des variantes comme :
      TickTick: ...
      TickTick_ ...
      espaces, underscores, ponctuation, etc.
    """
    return "".join(ch.lower() for ch in path if ch.isalnum())


def build_file_resolver(file_to_chunks: dict[str, list[dict]]) -> dict[str, str]:
    """
    Construit une table de résolution :
      clé candidate -> chemin canonique réel dans le corpus.

    Le chemin canonique est celui utilisé dans corpus.jsonl :
      task/filename.txt
    """
    resolver: dict[str, str] = {}

    for canonical_path in file_to_chunks.keys():
        canonical_path = canonical_path.replace("\\", "/")
        path_obj = Path(canonical_path)
        task_prefix = canonical_path.split("/", 1)[0] if "/" in canonical_path else ""

        candidates = set()

        candidates.add(canonical_path)
        candidates.add(canonical_path.replace("||", "__"))
        candidates.add(path_obj.name)

        if task_prefix:
            candidates.add(f"{task_prefix}/{path_obj.name}")

        # Variantes fréquentes observées dans LegalBench-RAG.
        candidates.add(canonical_path.replace(":", "_"))
        candidates.add(canonical_path.replace("_", ":"))
        candidates.add(canonical_path.replace("__", "||"))
        candidates.add(canonical_path.replace("||", "__"))

        # Clés normalisées.
        candidates.add(normalize_path_key(canonical_path))
        candidates.add(normalize_path_key(path_obj.name))

        for candidate in candidates:
            if candidate and candidate not in resolver:
                resolver[candidate] = canonical_path

    return resolver


def resolve_file_path(
    raw_file_path: str,
    file_resolver: dict[str, str],
) -> str | None:
    """
    Essaie de résoudre un file_path du benchmark vers un vrai fichier corpus.

    LegalBench-RAG contient parfois des chemins légèrement différents :
      - ':' vs '_'
      - '||' vs '__'
      - chemins avec ancien nom PDF + nom TXT : A.pdf||B.txt
    """
    if not isinstance(raw_file_path, str):
        return None

    raw = raw_file_path.replace("\\", "/")
    task_prefix = raw.split("/", 1)[0] if "/" in raw else ""

    direct_candidates = [
        raw,
        raw.replace("||", "__"),
        raw.replace(":", "_"),
        raw.replace("_", ":"),
    ]

    # Cas MAUD fréquent : task/A.pdf||B.txt.
    # Le vrai fichier peut être task/A.pdf__B.txt, task/B.txt,
    # ou une variante normalisée.
    if "||" in raw:
        before, after = raw.split("||", 1)

        direct_candidates.extend(
            [
                after,
                after.replace(":", "_"),
                after.replace("_", ":"),
                raw.replace("||", "__"),
                before + "__" + after,
            ]
        )

        if task_prefix:
            direct_candidates.extend(
                [
                    f"{task_prefix}/{after}",
                    f"{task_prefix}/{after.replace(':', '_')}",
                    f"{task_prefix}/{after.replace('_', ':')}",
                ]
            )

    # Matching direct.
    for candidate in direct_candidates:
        if candidate in file_resolver:
            return file_resolver[candidate]

    # Matching normalisé.
    for candidate in direct_candidates:
        normalized = normalize_path_key(candidate)
        if normalized in file_resolver:
            return file_resolver[normalized]

    return None


def select_items_for_task(
    *,
    task: str,
    items: list[dict],
    max_queries_per_task: int,
    seed: int,
) -> list[tuple[int, dict]]:
    """
    Sélectionne les exemples pour la mini-expérience.

    max_queries_per_task = 0 signifie full mode.

    On utilise une seed dérivée par tâche :
        task_seed = seed + somme des caractères du nom de tâche

    Cela garantit :
      - une sélection reproductible ;
      - une sélection différente pour chaque tâche ;
      - un seul paramètre seed global.
    """
    indexed_items = []

    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        query = item.get("query")
        snippets = item.get("snippets", [])

        if not isinstance(query, str) or not query.strip():
            continue

        if not isinstance(snippets, list) or len(snippets) == 0:
            continue

        indexed_items.append((idx, item))

    if max_queries_per_task <= 0:
        return indexed_items

    if len(indexed_items) <= max_queries_per_task:
        return indexed_items

    task_seed = seed + sum(ord(c) for c in task)
    rng = random.Random(task_seed)

    selected = rng.sample(indexed_items, max_queries_per_task)
    selected = sorted(selected, key=lambda x: x[0])

    return selected


def build_corpus(
    *,
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict], dict[str, list[dict]], dict[str, str]]:
    corpus_rows = []
    file_to_chunks = defaultdict(list)
    file_to_text = {}

    for task in TASKS:
        task_dir = CORPUS_DIR / task

        if not task_dir.exists():
            raise FileNotFoundError(f"Missing corpus task directory: {task_dir}")

        files = sorted([p for p in task_dir.rglob("*") if p.is_file()])

        print(f"Task {task}: reading {len(files)} corpus files")

        for path in files:
            rel_path = path.relative_to(CORPUS_DIR).as_posix()
            text = read_text_file(path)

            chunks = chunk_document(
                task=task,
                file_path=rel_path,
                text=text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            corpus_rows.extend(chunks)
            file_to_chunks[rel_path].extend(chunks)
            file_to_text[rel_path] = text

    return corpus_rows, dict(file_to_chunks), file_to_text


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def validate_answer_span(
    *,
    answer: str,
    extracted: str,
    counter: Counter,
) -> None:
    """
    Validation légère entre answer et texte extrait par span.

    On ne bloque pas le pipeline si ça ne matche pas parfaitement,
    parce que des différences d'espaces ou de nettoyage peuvent exister.
    On trace simplement le résultat dans metadata.json.
    """
    normalized_answer = normalize_space(answer)
    normalized_extracted = normalize_space(extracted)

    if normalized_answer and normalized_extracted:
        if normalized_answer[:120] in normalized_extracted:
            counter["answer_prefix_matches"] += 1
        elif normalized_extracted[:120] in normalized_answer:
            counter["extracted_prefix_matches"] += 1
        else:
            counter["mismatch_or_whitespace_shift"] += 1
    else:
        counter["empty_answer_or_extracted"] += 1


def build_queries_and_qrels(
    *,
    file_to_chunks: dict[str, list[dict]],
    file_to_text: dict[str, str],
    file_resolver: dict[str, str],
    max_queries_per_task: int,
    seed: int,
    min_overlap_chars: int,
) -> tuple[list[dict], list[tuple[str, str, str, int]], list[dict], dict]:
    """
    Construit queries.jsonl, qrels.tsv et qrels_detail.tsv.

    qrels format :
      query_id  0  doc_id  1
    """
    queries = []
    qrels_set = set()
    qrels_detail = []

    summary = {
        "tasks": {},
        "missing_files": [],
        "resolved_file_path_changes": [],
        "snippets_without_matching_chunk": [],
        "answer_span_validation": Counter(),
    }

    for task in TASKS:
        benchmark_path = BENCHMARKS_DIR / f"{task}.json"

        if not benchmark_path.exists():
            raise FileNotFoundError(f"Missing benchmark file: {benchmark_path}")

        data = load_json(benchmark_path)
        items = get_json_items(data)

        selected_items = select_items_for_task(
            task=task,
            items=items,
            max_queries_per_task=max_queries_per_task,
            seed=seed,
        )

        print(
            f"Task {task}: selected {len(selected_items)} queries "
            f"out of {len(items)} detected items"
        )

        task_num_snippets = 0
        task_qrels_set = set()
        task_qrels_detail_count = 0
        task_missing_files = 0
        task_resolved_changes = 0

        for original_index, item in selected_items:
            query_id = f"{task}_{original_index:05d}"
            query_text = item["query"].strip()
            snippets = item.get("snippets", [])

            queries.append(
                {
                    "query_id": query_id,
                    "task": task,
                    "original_index": original_index,
                    "text": query_text,
                    "num_snippets": len(snippets),
                }
            )

            for snippet_index, snippet in enumerate(snippets):
                if not isinstance(snippet, dict):
                    continue

                raw_file_path = snippet.get("file_path")
                span = snippet.get("span")
                answer = snippet.get("answer", "")

                if not isinstance(raw_file_path, str):
                    continue

                raw_file_path = raw_file_path.replace("\\", "/")
                resolved_file_path = resolve_file_path(raw_file_path, file_resolver)

                if resolved_file_path is None:
                    task_missing_files += 1
                    summary["missing_files"].append(
                        {
                            "task": task,
                            "query_id": query_id,
                            "snippet_index": snippet_index,
                            "file_path": raw_file_path,
                        }
                    )
                    continue

                if resolved_file_path != raw_file_path:
                    task_resolved_changes += 1
                    summary["resolved_file_path_changes"].append(
                        {
                            "task": task,
                            "query_id": query_id,
                            "snippet_index": snippet_index,
                            "raw_file_path": raw_file_path,
                            "resolved_file_path": resolved_file_path,
                        }
                    )

                file_path = resolved_file_path

                if file_path not in file_to_chunks:
                    task_missing_files += 1
                    summary["missing_files"].append(
                        {
                            "task": task,
                            "query_id": query_id,
                            "snippet_index": snippet_index,
                            "file_path": raw_file_path,
                            "resolved_file_path": resolved_file_path,
                            "reason": "resolved_path_not_in_file_to_chunks",
                        }
                    )
                    continue

                if (
                    not isinstance(span, list)
                    or len(span) != 2
                    or not isinstance(span[0], int)
                    or not isinstance(span[1], int)
                ):
                    continue

                span_start, span_end = int(span[0]), int(span[1])

                if span_end <= span_start:
                    continue

                task_num_snippets += 1

                full_text = file_to_text.get(file_path, "")
                extracted = full_text[span_start:span_end]

                validate_answer_span(
                    answer=answer,
                    extracted=extracted,
                    counter=summary["answer_span_validation"],
                )

                answer_len = max(1, span_end - span_start)

                # Si le span est court, on exige tout le span.
                # Sinon, on exige au moins min_overlap_chars.
                required_overlap = min(min_overlap_chars, answer_len)

                matched_any_chunk = False

                for chunk in file_to_chunks[file_path]:
                    ov = overlap_length(
                        span_start,
                        span_end,
                        int(chunk["char_start"]),
                        int(chunk["char_end"]),
                    )

                    if ov < required_overlap:
                        continue

                    matched_any_chunk = True

                    qrel = (query_id, "0", chunk["doc_id"], 1)
                    qrels_set.add(qrel)
                    task_qrels_set.add(qrel)

                    qrels_detail.append(
                        {
                            "query_id": query_id,
                            "task": task,
                            "snippet_index": snippet_index,
                            "raw_file_path": raw_file_path,
                            "resolved_file_path": resolved_file_path,
                            "file_path": file_path,
                            "span_start": span_start,
                            "span_end": span_end,
                            "answer_length": answer_len,
                            "required_overlap_chars": required_overlap,
                            "doc_id": chunk["doc_id"],
                            "chunk_start": chunk["char_start"],
                            "chunk_end": chunk["char_end"],
                            "overlap_chars": ov,
                            "overlap_fraction_of_answer": ov / answer_len,
                        }
                    )

                    task_qrels_detail_count += 1

                if not matched_any_chunk:
                    summary["snippets_without_matching_chunk"].append(
                        {
                            "task": task,
                            "query_id": query_id,
                            "snippet_index": snippet_index,
                            "file_path": file_path,
                            "raw_file_path": raw_file_path,
                            "resolved_file_path": resolved_file_path,
                            "span_start": span_start,
                            "span_end": span_end,
                            "answer_length": answer_len,
                            "required_overlap_chars": required_overlap,
                        }
                    )

        summary["tasks"][task] = {
            "detected_items": len(items),
            "selected_queries": len(selected_items),
            "snippets_in_selected_queries": task_num_snippets,
            "qrels": len(task_qrels_set),
            "qrels_detail_rows": task_qrels_detail_count,
            "missing_files": task_missing_files,
            "resolved_file_path_changes": task_resolved_changes,
        }

    qrels = sorted(qrels_set, key=lambda x: (x[0], x[2]))

    return queries, qrels, qrels_detail, summary


def write_qrels(path: Path, qrels: list[tuple[str, str, str, int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for query_id, iteration, doc_id, relevance in qrels:
            writer.writerow([query_id, iteration, doc_id, relevance])


def write_qrels_detail(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=QRELS_DETAIL_FIELDS,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            clean_row = dict(row)
            clean_row["overlap_fraction_of_answer"] = (
                f"{float(clean_row['overlap_fraction_of_answer']):.6f}"
            )
            writer.writerow(clean_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LegalBench-RAG mini as retrieval-ready JSONL/TSV files."
    )

    parser.add_argument(
        "--max-queries-per-task",
        type=int,
        default=50,
        help=(
            "Number of queries sampled per task. "
            "Use 0 for full benchmark."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Chunk overlap in characters.",
    )

    parser.add_argument(
        "--min-overlap-chars",
        type=int,
        default=50,
        help=(
            "Minimum character overlap required between a gold span and a chunk. "
            "For spans shorter than this, the full span length is required."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for mini sampling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("Preparing LegalBench-RAG mini")
    print("=" * 80)

    print(f"Raw directory: {RAW_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"max_queries_per_task = {args.max_queries_per_task}")
    print(f"chunk_size = {args.chunk_size}")
    print(f"overlap = {args.overlap}")
    print(f"min_overlap_chars = {args.min_overlap_chars}")
    print(f"seed = {args.seed}")

    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Missing corpus directory: {CORPUS_DIR}")

    if not BENCHMARKS_DIR.exists():
        raise FileNotFoundError(f"Missing benchmarks directory: {BENCHMARKS_DIR}")

    print("\nBuilding chunked corpus...")
    corpus_rows, file_to_chunks, file_to_text = build_corpus(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    print(f"Total chunks: {len(corpus_rows)}")
    print(f"Total source files: {len(file_to_chunks)}")

    file_resolver = build_file_resolver(file_to_chunks)
    print(f"File resolver entries: {len(file_resolver)}")

    print("\nBuilding queries and qrels...")
    queries, qrels, qrels_detail, summary = build_queries_and_qrels(
        file_to_chunks=file_to_chunks,
        file_to_text=file_to_text,
        file_resolver=file_resolver,
        max_queries_per_task=args.max_queries_per_task,
        seed=args.seed,
        min_overlap_chars=args.min_overlap_chars,
    )

    corpus_path = OUT_DIR / "corpus.jsonl"
    queries_path = OUT_DIR / "queries.jsonl"
    qrels_path = OUT_DIR / "qrels.tsv"
    qrels_detail_path = OUT_DIR / "qrels_detail.tsv"
    metadata_path = OUT_DIR / "metadata.json"

    print("\nWriting output files...")
    write_jsonl(corpus_path, corpus_rows)
    write_jsonl(queries_path, queries)
    write_qrels(qrels_path, qrels)
    write_qrels_detail(qrels_detail_path, qrels_detail)

    metadata = {
        "name": "legalbench_rag_mini",
        "raw_dir": str(RAW_DIR),
        "output_dir": str(OUT_DIR),
        "tasks": TASKS,
        "max_queries_per_task": args.max_queries_per_task,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "min_overlap_chars": args.min_overlap_chars,
        "seed": args.seed,
        "num_corpus_chunks": len(corpus_rows),
        "num_source_files": len(file_to_chunks),
        "num_file_resolver_entries": len(file_resolver),
        "num_queries": len(queries),
        "num_qrels": len(qrels),
        "num_qrels_detail_rows": len(qrels_detail),
        "summary": {
            **summary,
            "answer_span_validation": dict(summary["answer_span_validation"]),
            "num_missing_files": len(summary["missing_files"]),
            "num_resolved_file_path_changes": len(summary["resolved_file_path_changes"]),
            "num_snippets_without_matching_chunk": len(
                summary["snippets_without_matching_chunk"]
            ),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved corpus to: {corpus_path}")
    print(f"Saved queries to: {queries_path}")
    print(f"Saved qrels to: {qrels_path}")
    print(f"Saved qrels detail to: {qrels_detail_path}")
    print(f"Saved metadata to: {metadata_path}")

    print("\n" + "=" * 80)
    print("Preparation summary")
    print("=" * 80)

    print(f"Corpus chunks: {len(corpus_rows)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
    print(f"Qrels detail rows: {len(qrels_detail)}")

    print("\nPer-task summary:")
    for task, stats in metadata["summary"]["tasks"].items():
        print(f"  {task}: {stats}")

    print("\nValidation:")
    print(f"  Missing files: {len(summary['missing_files'])}")
    print(f"  Resolved file path changes: {len(summary['resolved_file_path_changes'])}")
    print(
        "  Snippets without matching chunk: "
        f"{len(summary['snippets_without_matching_chunk'])}"
    )
    print(f"  Answer/span validation: {dict(summary['answer_span_validation'])}")

    print("\nNext script:")
    print("  scripts/legalbench/04_bm25_legalbench_original.py")


if __name__ == "__main__":
    main()
