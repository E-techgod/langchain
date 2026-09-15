import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

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


@tool
def search_promotions(query: str, runtime: ToolRuntime) -> str:
    """Search saved promotions by meaning, benefit, provider, date, or restriction."""
    matches = runtime.store.search(
        PROMOTION_NAMESPACE,
        query=query,
        limit=5,
    )
    results = [
        {
            "score": match.score,
            **{key: value for key, value in match.value.items() if key != "text"},
        }
        for match in matches
    ]
    return json.dumps(results, ensure_ascii=False)


def build_agent(store: PostgresStore, checkpointer: PostgresSaver):
    model = ChatGroq(
        model=os.getenv("PROMOTION_AGENT_MODEL", "openai/gpt-oss-120b"),
        temperature=0.0,
        groq_api_key=required_env("GROQ_API_KEY"),
    )
    return create_agent(
        model=model,
        tools=[search_promotions],
        store=store,
        checkpointer=checkpointer,
        system_prompt=(
            "You are a promotion assistant. Before answering any question about "
            "promotions, always call search_promotions with a concise semantic "
            "query. Ground the answer only in the returned promotion memories. "
            "Include relevant dates and restrictions. Give a concise answer, with no more than 5 bullet points. "
            "Straightforwardly say if no matching promotion was found. Do not make up any information."
            "If the search does not contain the answer, say that no matching promotion was found."
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync promotion memory and chat with the promotions agent."
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Load promotion_results.json into pgvector and exit.",
    )
    parser.add_argument(
        "--thread-id",
        default="promotion-cli",
        help="Conversation ID used by the PostgreSQL checkpointer.",
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
    ) as store, PostgresSaver.from_conn_string(database_url) as checkpointer:
        store.setup()
        checkpointer.setup()
        synced = sync_promotion_memories(store)
        print(f"Synced {synced} promotion memories to PostgreSQL/pgvector.")

        if args.sync_only:
            return

        agent = build_agent(store, checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

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

            response = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
            print(f"Agent: {response['messages'][-1].content}")


if __name__ == "__main__":
    main()
