import argparse
import json
import os
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
        memory["text"] = searchable_text(promotion)
        store.put(PROMOTION_NAMESPACE, promotion.filename, memory)

    return len(raw_results)


def top_relevance_score(store: PostgresStore, query: str) -> float | None:
    matches = store.search(PROMOTION_NAMESPACE, query=query, limit=1)
    return matches[0].score if matches else None


template = """"You are a promotion assistant. Before answering any question about 
            promotions, always call search_promotions with a concise semantic 
            query. Ground the answer only in the returned promotion memories. 
            Include relevant dates and restrictions. Give a concise answer, with no more than 5 bullet points. 
            Straightforwardly say if no matching promotion was found. Do not make up any information.
            If the search does not contain the answer, say "I cannot answer this, due to unrelated query"

Context:
{context}

Question:
{question}
"""

def retrieve_promotions(store: PostgresStore, query: str) -> list[dict[str, Any]]:
    matches = store.search(PROMOTION_NAMESPACE, query=query, limit=5)
    return [
        {
            "score": match.score,
            **{key: value for key, value in match.value.items() if key != "text"},
        }
        for match in matches
        if match.score is not None and match.score >= relevance_threshold()
    ]


def build_chain(store: PostgresStore):
    model = ChatGroq(
        model=os.getenv("PROMOTION_AGENT_MODEL", "openai/gpt-oss-120b"),
        temperature=0.0,
        groq_api_key=required_env("GROQ_API_KEY"),
    )
    prompt = ChatPromptTemplate.from_template(template.replace(
        "Before answering promotion queries, call `search_promotions` with a concise semantic query. ",
        "",
    ))
    return RunnableLambda(
        lambda question: {
            "context": json.dumps(retrieve_promotions(store, question), ensure_ascii=False),
            "question": question,
        }
    ) | prompt | model | StrOutputParser()

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

        chain = build_chain(store)

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

            response = chain.invoke(question)
            print(f"Agent: {response}")


if __name__ == "__main__":
    main()
