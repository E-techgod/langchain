import argparse
import calendar
import json
import os
import re
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.postgres import PostgresStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from promotion_schema import Promotion


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "promotion_results.json"
PROMOTION_NAMESPACE = ("promotions",)
EMBEDDING_DIMENSIONS = 768


class PromotionMemory(Promotion):
    filename: str
    Promo_info: str | list[str]


def required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value

################## Handling relevance threshold and searchable text for promotions ##################
def relevance_threshold() -> float:
    return float(os.getenv("PROMOTION_RELEVANCE_THRESHOLD", "0.30"))

def searchable_text(promotion: PromotionMemory) -> str:
    restrictions = promotion.Restrictions or ["None listed"]
    promotion_info = (
        promotion.Promo_info
        if isinstance(promotion.Promo_info, str)
        else "; ".join(promotion.Promo_info)
    )


SPANISH_MONTHS = {
    "enero": 1,
    "febrero": 2,
    "marzo": 3,
    "abril": 4,
    "mayo": 5,
    "junio": 6,
    "julio": 7,
    "agosto": 8,
    "septiembre": 9,
    "setiembre": 9,
    "octubre": 10,
    "noviembre": 11,
    "diciembre": 12,
}


def promotion_end_date(value: str) -> date | None:
    normalized = value.lower().strip()
    match = re.search(
        r"(?:del?\s+)?(?:\d{1,2}\s+al\s+)?(\d{1,2})\s+de\s+([a-z]+)\s+de\s+(\d{4})",
        normalized,
    )
    if match:
        day, month_name, year = match.groups()
        month = SPANISH_MONTHS.get(month_name)
        if month:
            return date(int(year), month, int(day))

    month_match = re.search(r"([a-z]+)\s+(\d{4})", normalized)
    if month_match:
        month = SPANISH_MONTHS.get(month_match.group(1))
        if month:
            year = int(month_match.group(2))
            return date(year, month, calendar.monthrange(year, month)[1])

    return None


def live_filter() -> dict[str, dict[str, str]]:
    return {"validation_date": {"$gte": date.today().isoformat()}}
    return "\n".join(
        [
            f"Insurance provider: {promotion.Insurance_Provided}",
            f"Promotion date: {promotion.Promo_date}",
            f"Promotion information: {promotion_info}",
            f"Restrictions: {'; '.join(restrictions)}",
        ]
    )


def sync_promotion_memories(store: PostgresStore) -> int:
    """Validate and upsert every promotion from the JSON results file."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"{RESULTS_FILE.name} does not exist. Run `uv run image.py` first."
        )

    raw_results: Any = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    if not isinstance(raw_results, list):
        raise ValueError(f"{RESULTS_FILE.name} must contain a JSON array.")

    for raw_promotion in raw_results:
        promotion = PromotionMemory.model_validate(raw_promotion)
        memory = promotion.model_dump()
        end_date = promotion_end_date(promotion.Promo_date)
        memory["validation_date"] = end_date.isoformat() if end_date else None
        memory["text"] = searchable_text(promotion)
        store.put(PROMOTION_NAMESPACE, promotion.filename, memory)

    return len(raw_results)


def top_relevance_score(store: PostgresStore, query: str) -> float | None:
    matches = store.search(
        PROMOTION_NAMESPACE,
        query=query,
        filter=live_filter(),
        limit=1,
    )
    return matches[0].score if matches else None


template = """You are a promotion assistant. Before answering any question about 
            promotions, always call search_promotions with a concise semantic 
            query. Ground the answer only in the returned promotion memories. 
            Include relevant dates and restrictions. Give a concise answer, keep it simple. 

Context:
{context}

Question:
{question}
"""

relevance_template = """Determine whether the promotion context is relevant to the user question.
Return only YES or NO.

Promotion context:
{context}

User question:
{question}
"""

def retrieve_promotions(store: PostgresStore, query: str) -> list[dict[str, Any]]:
    matches = store.search(
        PROMOTION_NAMESPACE,
        query=query,
        filter=live_filter(),
        limit=50,
    )
    return [
        {
            "score": match.score,
            **{key: value for key, value in match.value.items() if key != "text"},
        }
        for match in matches
        if match.score is not None and match.score >= relevance_threshold()
    ]


def build_chains(store: PostgresStore):
    relevance_model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.0,
        groq_api_key=required_env("GROQ_API_KEY"),
    )
    answer_model = ChatGroq(
        model=os.getenv("PROMOTION_ANSWER_MODEL", "openai/gpt-oss-120b"),
        temperature=0.0,
        groq_api_key=required_env("GROQ_API_KEY"),
    )
    answer_prompt = ChatPromptTemplate.from_template(template.replace(
        "Before answering promotion queries, call `search_promotions` with a concise semantic query. ",
        "",
    ))
    relevance_prompt = ChatPromptTemplate.from_template(relevance_template)
    context_and_question = RunnableLambda(
        lambda question: {
            "context": json.dumps(retrieve_promotions(store, question), ensure_ascii=False),
            "question": question,
        }
    )
    relevance_chain = context_and_question | relevance_prompt | relevance_model | StrOutputParser()
    answer_chain = context_and_question | answer_prompt | answer_model | StrOutputParser()
    return relevance_chain, answer_chain

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync promotion memory and chat with the promotions agent."
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Load promotion_results.json into pgvector and exit.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    database_url = required_env("DATABASE_URL")
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    )

    index_config = {
        "dims": EMBEDDING_DIMENSIONS,
        "embed": embeddings,
        "fields": ["text"],
    }

    with PostgresStore.from_conn_string(
        database_url,
        index=index_config,
    ) as store:
        store.setup()
        synced = sync_promotion_memories(store)
        print(f"Synced {synced} promotion memories to PostgreSQL/pgvector.")

        if args.sync_only:
            return

        relevance_chain, answer_chain = build_chains(store)

        print("Ask about promotions. Type `exit` to stop.")
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue

            score = top_relevance_score(store, question)
            print(f"Similarity score: {score if score is not None else 'no match'}")

            if score is None or score < relevance_threshold():
                print("Agent: I cannot answer this based on the available information.")
                continue

            relevance = relevance_chain.invoke(question).strip().upper()
            print(f"OSS relevance check: {relevance}")
            if not relevance.startswith("YES"):
                print("Agent: I cannot answer this based on the available information.")
                continue

            response = answer_chain.invoke(question)
            print(f"Agent: {response}")


if __name__ == "__main__":
    main()
