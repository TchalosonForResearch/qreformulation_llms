import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm


DEFAULT_MODEL = os.getenv("OPENAI_REFORMULATION_MODEL", "gpt-5.5")
PROMPT_VERSION = "bsard_reformulation_v2"

FORBIDDEN_EXPRESSIONS = [
    "loi française",
    "droit français",
    "tribunal judiciaire",
    "juge aux affaires familiales",
    "aide juridictionnelle",
    "prestation compensatoire",
    "code pénal français",
    "code civil français",
]

HYDE_ALLOWED_PREFIXES = [
    "Un article pertinent pourrait traiter",
    "Le texte recherché pourrait préciser",
    "La disposition applicable pourrait concerner",
]

REFORMULATION_FORMAT = {
    "type": "json_schema",
    "name": "bsard_query_reformulation",
    "description": "Trois reformulations françaises d'une requête BSARD.",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "legal_rewrite": {
                "type": "string",
                "description": "Reformulation juridique interrogative équivalente à la question originale.",
            },
            "keyword_expansion": {
                "type": "string",
                "description": "Liste courte de mots-clés séparés par des virgules.",
            },
            "hyde_style": {
                "type": "string",
                "description": "Passage hypothétique prudent décrivant le contenu qu'un article pertinent pourrait contenir.",
            },
        },
        "required": ["legal_rewrite", "keyword_expansion", "hyde_style"],
        "additionalProperties": False,
    },
}

SYSTEM_INSTRUCTIONS = (
    "Tu es un assistant spécialisé en reformulation de requêtes pour le retrieval "
    "juridique belge francophone. Tu dois préserver strictement le besoin "
    "informationnel original, éviter toute dérive vers le droit français et produire "
    "uniquement la structure JSON demandée."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                ids.add(str(row["query_id"]))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"Invalid existing output on line {line_number} of {path}: {exc}"
                ) from exc
    return ids


def safe_json_loads(text: str | None) -> dict[str, Any]:
    if text is None or not text.strip():
        raise ValueError("Empty model response")

    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("The model response must be a JSON object")
    return parsed


def contains_forbidden_expression(text: str) -> str | None:
    text_lower = text.lower()
    for expression in FORBIDDEN_EXPRESSIONS:
        if expression.lower() in text_lower:
            return expression
    return None


def starts_with_allowed_hyde_prefix(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in HYDE_ALLOWED_PREFIXES)


def build_prompt(query: dict[str, Any]) -> str:
    question = str(query["question"]).strip()
    extra_description = query.get("extra_description")
    extra_text = (
        "Aucun contexte additionnel."
        if extra_description is None or not str(extra_description).strip()
        else str(extra_description).strip()
    )

    return f"""
Tu dois reformuler une question juridique française issue d'un benchmark de retrieval d'articles de loi belges francophones.

CONTEXTE IMPORTANT :
- Le contexte juridique est belge francophone.
- Le contexte n'est pas le droit français.
- Tu dois éviter toute terminologie institutionnelle propre au droit français.
- Tu dois préserver exactement le besoin informationnel original.

Question originale :
{question}

Contexte additionnel éventuel :
{extra_text}

Génère les trois champs suivants :
- "legal_rewrite"
- "keyword_expansion"
- "hyde_style"

CONTRAINTES GÉNÉRALES :
- Toutes les sorties doivent être en français.
- Préserve strictement le même besoin informationnel que la question originale.
- Ne réponds jamais directement à la question.
- Ne cite aucun numéro d'article.
- N'ajoute aucune condition juridique absente.
- Ne rends pas la question plus générale ou plus spécifique.
- Ne change pas les personnes, institutions, temporalités, faits, régions ou relations juridiques.
- Préserve les mentions régionales présentes dans la question : Wallonie, Bruxelles, Belgique, etc.
- Ne mentionne pas le nom du benchmark.
- N'invente pas de dates, délais, montants, sanctions, seuils, conditions ou procédures.
- N'utilise pas de vocabulaire institutionnel français si ce terme n'est pas présent dans la question originale.

EXPRESSIONS STRICTEMENT INTERDITES :
- loi française
- droit français
- tribunal judiciaire
- juge aux affaires familiales
- aide juridictionnelle
- prestation compensatoire
- code pénal français
- code civil français

1. "legal_rewrite" :
Reformule la question dans un registre plus juridique, mais sous forme de question.
La reformulation doit rester équivalente à la question originale.
Elle ne doit pas introduire de nouvelle notion juridique absente.
Elle ne doit pas remplacer une relation non mariée par une relation mariée.
Elle ne doit pas remplacer une institution belge par une institution française.
Elle ne doit pas répondre à la question.

2. "keyword_expansion" :
Donne une courte liste de mots-clés ou expressions utiles au retrieval juridique.
La valeur doit être une seule chaîne de texte, avec les mots-clés séparés par des virgules.
Les mots-clés doivent rester proches de la question originale.
Ne rajoute pas de domaine juridique absent.
N'utilise pas de termes français incompatibles avec le contexte belge.
Ne cite aucun article.

3. "hyde_style" :
Rédige un court passage hypothétique destiné à aider un système de retrieval.
Le passage doit décrire le type de contenu qu'un article pertinent pourrait contenir.
Il ne doit pas répondre à la question.
Il ne doit pas affirmer une règle juridique comme certaine.
Il ne doit pas inventer de règle, date, sanction, délai, seuil ou condition.
Il doit rester prudent et non assertif.

Le champ "hyde_style" doit obligatoirement commencer par l'une de ces formulations :
- "Un article pertinent pourrait traiter..."
- "Le texte recherché pourrait préciser..."
- "La disposition applicable pourrait concerner..."

Exemples de style HyDE accepté :
- "Un article pertinent pourrait traiter des conditions applicables à..."
- "Le texte recherché pourrait préciser les règles relatives à..."
- "La disposition applicable pourrait concerner les effets juridiques de..."

Exemples de style HyDE refusé :
- "La loi prévoit que..."
- "Le délai commence à courir..."
- "Le propriétaire peut..."
- "La victime a droit à..."
- "La procédure entraîne..."
""".strip()


def build_repair_instruction(last_error: str) -> str:
    return f"""
ATTENTION : ta sortie précédente a été refusée pour la raison suivante :
{last_error}

Corrige strictement ce problème.

Rappels obligatoires :
- Le champ "hyde_style" doit commencer par "Un article pertinent pourrait traiter...", "Le texte recherché pourrait préciser..." ou "La disposition applicable pourrait concerner...".
- Ne commence jamais "hyde_style" par une affirmation directe.
- Ne réponds pas à la question.
- N'utilise aucune terminologie du droit français.
- N'invente pas de date, délai, sanction, condition ou seuil.
- Retourne uniquement les trois champs demandés.
""".strip()


def validate_output(obj: dict[str, Any]) -> dict[str, str]:
    required_keys = ["legal_rewrite", "keyword_expansion", "hyde_style"]

    if set(obj) != set(required_keys):
        missing = sorted(set(required_keys) - set(obj))
        extra = sorted(set(obj) - set(required_keys))
        raise ValueError(f"Unexpected JSON keys; missing={missing}, extra={extra}")

    cleaned: dict[str, str] = {}
    for key in required_keys:
        value = obj[key]
        if not isinstance(value, str):
            raise ValueError(f"Key {key} must be a string")
        value = value.strip()
        if not value:
            raise ValueError(f"Key {key} is empty")
        cleaned[key] = value

    for key, value in cleaned.items():
        forbidden = contains_forbidden_expression(value)
        if forbidden:
            raise ValueError(f"Forbidden expression found in {key}: {forbidden}")

    if not starts_with_allowed_hyde_prefix(cleaned["hyde_style"]):
        raise ValueError(
            "hyde_style must start with one of the required cautious formulations"
        )

    return cleaned


def object_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return str(value)


def generate_one(
    client: OpenAI,
    model: str,
    query: dict[str, Any],
    reasoning_effort: str,
    max_output_tokens: int,
    max_retries: int,
) -> dict[str, Any]:
    base_prompt = build_prompt(query)
    last_error: str | None = None
    last_content: str | None = None
    last_response: Any = None

    for attempt in range(1, max_retries + 1):
        user_prompt = (
            base_prompt
            if last_error is None
            else base_prompt + "\n\n" + build_repair_instruction(last_error)
        )

        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=user_prompt,
                reasoning={"effort": reasoning_effort},
                max_output_tokens=max_output_tokens,
                text={
                    "format": REFORMULATION_FORMAT,
                    "verbosity": "low",
                },
                store=False,
            )
            last_response = response
            content = response.output_text
            last_content = content

            if getattr(response, "status", "completed") != "completed":
                details = object_to_dict(getattr(response, "incomplete_details", None))
                raise ValueError(f"Incomplete API response: {details}")

            validated = validate_output(safe_json_loads(content))
            return {
                "ok": True,
                "data": validated,
                "attempts": attempt,
                "raw_output": content,
                "response_id": getattr(response, "id", None),
                "resolved_model": getattr(response, "model", model),
                "usage": object_to_dict(getattr(response, "usage", None)),
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            query_id = query.get("query_id", "unknown")
            print(
                f"\nError for query {query_id} attempt "
                f"{attempt}/{max_retries}: {last_error}"
            )
            if attempt < max_retries:
                time.sleep(min(30.0, 2.0 ** attempt))

    return {
        "ok": False,
        "error": last_error,
        "attempts": max_retries,
        "raw_output": last_content,
        "response_id": getattr(last_response, "id", None),
        "resolved_model": getattr(last_response, "model", model),
        "usage": object_to_dict(getattr(last_response, "usage", None)),
    }


def make_base_row(query: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    question = str(query["question"]).strip()
    original = str(query.get("text") or question).strip()
    return {
        "query_id": str(query["query_id"]),
        "original": original,
        "question": question,
        "extra_description": query.get("extra_description"),
        "generator": args.model,
        "provider": "openai",
        "generator_model": args.model,
        "api_endpoint": "responses",
        "prompt_version": PROMPT_VERSION,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def validate_query(query: dict[str, Any], index: int) -> None:
    for key in ("query_id", "question"):
        if key not in query:
            raise ValueError(f"Input row {index} is missing required key: {key}")
    if not str(query["question"]).strip():
        raise ValueError(f"Input row {index} has an empty question")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/queries_test.jsonl")
    parser.add_argument(
        "--output",
        default="data/reformulated_queries_test_gpt55_v2.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default="low",
    )
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_name(output_path.stem + "_errors.jsonl")

    queries = load_jsonl(input_path)
    for index, query in enumerate(queries, start=1):
        validate_query(query, index)

    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        queries = queries[: args.limit]

    if args.dry_run:
        if not queries:
            raise ValueError("No query available for dry-run")
        print(build_prompt(queries[0]))
        return

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. In PowerShell: "
            "$env:OPENAI_API_KEY=\"your_key_here\""
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(output_path)

    client = OpenAI(
        api_key=api_key,
        timeout=args.timeout,
        max_retries=0,
    )

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Error file: {error_path}")
    print(f"Model: {args.model}")
    print(f"Prompt version: {PROMPT_VERSION}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Queries selected: {len(queries)}")
    print(f"Already processed in output: {len(existing_ids)}")

    valid_count = 0
    error_count = 0
    skipped_count = 0

    for query in tqdm(queries):
        query_id = str(query["query_id"])
        if query_id in existing_ids:
            skipped_count += 1
            continue

        result = generate_one(
            client=client,
            model=args.model,
            query=query,
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.max_retries,
        )
        base_row = make_base_row(query, args)

        if result["ok"]:
            data = result["data"]
            row = {
                **base_row,
                "resolved_model": result["resolved_model"],
                "response_id": result["response_id"],
                "usage": result["usage"],
                "system_prompt_sha256": hashlib.sha256(
                    SYSTEM_INSTRUCTIONS.encode("utf-8")
                ).hexdigest(),
                "user_prompt_sha256": hashlib.sha256(
                    build_prompt(query).encode("utf-8")
                ).hexdigest(),
                "legal_rewrite": data["legal_rewrite"],
                "keyword_expansion": data["keyword_expansion"],
                "hyde_style": data["hyde_style"],
                "validation_status": "valid",
                "attempts": result["attempts"],
            }
            append_jsonl(output_path, row)
            existing_ids.add(query_id)
            valid_count += 1
        else:
            error_row = {
                **base_row,
                "resolved_model": result["resolved_model"],
                "response_id": result["response_id"],
                "usage": result["usage"],
                "validation_status": "invalid",
                "attempts": result["attempts"],
                "error": result["error"],
                "raw_output": result["raw_output"],
            }
            append_jsonl(error_path, error_row)
            error_count += 1
            print(f"\nSaved invalid output for query {query_id} to {error_path}")

        time.sleep(max(0.0, args.sleep))

    print("\nGeneration completed.")
    print("=" * 80)
    print(f"Valid outputs: {valid_count}")
    print(f"Invalid outputs: {error_count}")
    print(f"Skipped existing outputs: {skipped_count}")
    print(f"Saved valid outputs to: {output_path}")
    print(f"Saved invalid outputs to: {error_path}")


if __name__ == "__main__":
    main()
