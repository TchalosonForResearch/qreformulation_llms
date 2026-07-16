import argparse
import json
import os
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


DATA = Path("data")
OUT = Path("data")
OUT.mkdir(exist_ok=True)


DEFAULT_MODEL = os.getenv("DEEPSEEK_REFORMULATION_MODEL", "deepseek-v4-flash")
PROMPT_VERSION = "v2"


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


# On autorise des variantes grammaticales :
# "traiter de", "traiter des", "traiter du", "traiter d'une", etc.
HYDE_ALLOWED_PREFIXES = [
    "Un article pertinent pourrait traiter",
    "Le texte recherché pourrait préciser",
    "La disposition applicable pourrait concerner",
]


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_ids(path):
    if not path.exists():
        return set()

    ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            ids.add(str(row["query_id"]))

    return ids


def safe_json_loads(text):
    """
    DeepSeek JSON mode produit normalement du JSON valide.
    Cette fonction ajoute une sécurité si le modèle entoure le JSON avec du texte.
    """
    if text is None:
        raise ValueError("Empty model response")

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])

        raise


def contains_forbidden_expression(text):
    text_lower = text.lower()

    for expr in FORBIDDEN_EXPRESSIONS:
        if expr.lower() in text_lower:
            return expr

    return None


def starts_with_allowed_hyde_prefix(text):
    text = text.strip()

    return any(text.startswith(prefix) for prefix in HYDE_ALLOWED_PREFIXES)


def build_prompt(query):
    question = query["question"]
    extra_description = query.get("extra_description")

    if extra_description is None or str(extra_description).strip() == "":
        extra_text = "Aucun contexte additionnel."
    else:
        extra_text = str(extra_description).strip()

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

Génère exactement un objet JSON valide avec les clés suivantes :
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

DÉFINITION DES TROIS CHAMPS :

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

FORMAT OBLIGATOIRE :
{{
  "legal_rewrite": "...",
  "keyword_expansion": "...",
  "hyde_style": "..."
}}
""".strip()


def build_repair_instruction(last_error):
    return f"""

ATTENTION : ta sortie précédente a été refusée pour la raison suivante :
{last_error}

Corrige strictement ce problème.

Rappels obligatoires :
- Le champ "hyde_style" doit commencer par :
  "Un article pertinent pourrait traiter..."
  ou "Le texte recherché pourrait préciser..."
  ou "La disposition applicable pourrait concerner..."
- Ne commence jamais "hyde_style" par une affirmation directe.
- Ne réponds pas à la question.
- N'utilise aucune terminologie du droit français.
- N'invente pas de date, délai, sanction, condition ou seuil.
- Retourne uniquement un objet JSON valide.
""".strip()


def validate_output(obj):
    required_keys = ["legal_rewrite", "keyword_expansion", "hyde_style"]

    for key in required_keys:
        if key not in obj:
            raise ValueError(f"Missing key: {key}")

        if not isinstance(obj[key], str):
            raise ValueError(f"Key {key} must be a string")

        if not obj[key].strip():
            raise ValueError(f"Key {key} is empty")

    cleaned = {
        "legal_rewrite": obj["legal_rewrite"].strip(),
        "keyword_expansion": obj["keyword_expansion"].strip(),
        "hyde_style": obj["hyde_style"].strip(),
    }

    for key, value in cleaned.items():
        forbidden = contains_forbidden_expression(value)

        if forbidden:
            raise ValueError(
                f"Forbidden expression found in {key}: {forbidden}"
            )

    hyde = cleaned["hyde_style"]

    if not starts_with_allowed_hyde_prefix(hyde):
        raise ValueError(
            "hyde_style must start with one of the required cautious formulations."
        )

    return cleaned


def generate_one(client, model, query, max_retries=3):
    base_prompt = build_prompt(query)
    last_error = None
    last_content = None

    for attempt in range(1, max_retries + 1):
        try:
            if last_error is None:
                user_prompt = base_prompt
            else:
                user_prompt = base_prompt + "\n\n" + build_repair_instruction(last_error)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Tu es un assistant spécialisé en reformulation de requêtes "
                            "pour le retrieval juridique belge francophone. "
                            "Tu dois préserver strictement le besoin informationnel original, "
                            "éviter toute dérive vers le droit français, "
                            "et produire uniquement du JSON valide."
                        ),
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                stream=False,
                extra_body={
                    "thinking": {"type": "disabled"}
                },
            )

            content = response.choices[0].message.content
            last_content = content

            obj = safe_json_loads(content)
            validated = validate_output(obj)

            return {
                "ok": True,
                "data": validated,
                "attempts": attempt,
                "raw_output": content,
            }

        except Exception as e:
            last_error = str(e)

            print(
                f"\nError for query {query['query_id']} "
                f"attempt {attempt}/{max_retries}: {last_error}"
            )

            if attempt < max_retries:
                time.sleep(2 * attempt)

    return {
        "ok": False,
        "error": last_error,
        "attempts": max_retries,
        "raw_output": last_content,
    }


def make_base_row(query, args):
    return {
        "query_id": str(query["query_id"]),
        "original": query["text"],
        "question": query["question"],
        "extra_description": query.get("extra_description"),
        "generator": args.model,
        "prompt_version": PROMPT_VERSION,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/queries_test.jsonl")
    parser.add_argument(
        "--output",
        default="data/reformulated_queries_test_deepseek_v2.jsonl"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=3)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = output_path.with_name(output_path.stem + "_errors.jsonl")

    queries = load_jsonl(input_path)

    if args.limit is not None:
        queries = queries[: args.limit]

    existing_ids = load_existing_ids(output_path)

    # Reproductibilité et sécurité :
    # La clé doit être définie localement dans l'environnement.
    # PowerShell :
    # $env:DEEPSEEK_API_KEY="votre_cle"
    api_key = "sk-960e2ee87ce84c75bb96225532b69c15"

    if not api_key:
        raise EnvironmentError(
            "DEEPSEEK_API_KEY is not set. "
            "Use PowerShell: $env:DEEPSEEK_API_KEY=\"your_key_here\" "
            "or setx DEEPSEEK_API_KEY \"your_key_here\""
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Error file: {error_path}")
    print(f"Model: {args.model}")
    print(f"Prompt version: {PROMPT_VERSION}")
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
            max_retries=args.max_retries,
        )

        base_row = make_base_row(query, args)

        if result["ok"]:
            data = result["data"]

            row = {
                **base_row,
                "legal_rewrite": data["legal_rewrite"],
                "keyword_expansion": data["keyword_expansion"],
                "hyde_style": data["hyde_style"],
                "validation_status": "valid",
                "attempts": result["attempts"],
            }

            append_jsonl(output_path, row)
            valid_count += 1

        else:
            error_row = {
                **base_row,
                "validation_status": "invalid",
                "attempts": result["attempts"],
                "error": result["error"],
                "raw_output": result["raw_output"],
            }

            append_jsonl(error_path, error_row)
            error_count += 1

            print(f"\nSaved invalid output for query {query_id} to {error_path}")

        time.sleep(args.sleep)

    print("\nGeneration completed.")
    print("=" * 80)
    print(f"Valid outputs: {valid_count}")
    print(f"Invalid outputs: {error_count}")
    print(f"Skipped existing outputs: {skipped_count}")
    print(f"Saved valid outputs to: {output_path}")
    print(f"Saved invalid outputs to: {error_path}")


if __name__ == "__main__":
    main()