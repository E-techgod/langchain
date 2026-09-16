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

## Job application email workflow

`job_email.py` reads the newest unread inbox message through IMAP, classifies it
as `rejected`, `moving_forward`, or `other`, and only drafts a reply for a clear
invitation to the next step. It never sends the draft until a human approves the
`send_email` tool call. The inbox read does not mark the message as seen.

Set `GROQ_API_KEY` and the `EMAIL_*` settings shown in `.env.example`. Your email
provider may require an app password and IMAP/SMTP access. Then invoke the graph
from Python:

```python
from job_email import graph
from langgraph.types import Command

config = {"configurable": {"thread_id": "review-1"}}
result = graph.invoke({"messages": []}, config=config)
print(result.get("__interrupt__", result))  # Inspect the proposed send_email arguments.

# Only after reviewing the recipient and message:
# graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
# To decline, use {"type": "reject", "message": "Do not send"} instead.
```

`InMemorySaver` keeps a pending review only in the current Python process. Use a
persistent LangGraph checkpointer if review must survive a restart. Supply
`{"email": {...}}` in the input state to classify a selected message without
reading the inbox.
