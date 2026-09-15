# Promotion extraction and semantic-memory agent

`image.py` extracts structured promotion data from JPEG files in `promos/` and
writes it to `promotion_results.json`. `promotion_agent.py` validates those
results, upserts them into a LangGraph PostgreSQL store with a pgvector index,
and provides a retrieval tool to a Groq-powered agent.

## Setup

Copy `.env.example` to `.env` and provide your Groq and OpenAI API keys. Groq
powers chat and OpenAI creates the embeddings used for semantic search.

Start PostgreSQL with the pgvector extension:

```bash
docker compose up -d postgres
```

Extract promotions, then sync and run the agent:

```bash
uv run image.py
uv run promotion_agent.py --sync-only
uv run promotion_agent.py
```

The agent uses two persistent memory mechanisms:

- `PostgresStore` stores promotion facts and retrieves them semantically through
  pgvector.
- `PostgresSaver` checkpoints each conversation using `--thread-id`. Reusing a
  thread ID resumes that conversation; using a new ID starts a new conversation.

Every agent startup resynchronizes `promotion_results.json`. The source filename
is used as the memory key, so an updated record replaces the prior record instead
of creating a duplicate.
