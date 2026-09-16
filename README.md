# Promotion Agents

This project extracts promotion details from images, stores them in PostgreSQL
with pgvector, and answers promotion questions with a guarded retrieval flow.

## Setup

Install dependencies and copy `.env.example` to `.env`:

```bash
uv sync
```

Configure the API keys and database URL in `.env`. Do not commit `.env`.

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

`image.py` validates extracted promotions with the `Promotion` schema and
Guardrails before writing `promotion_results.json`. Its primary model is
Groq-backed structured output, with an LCEL fallback model if extraction fails.

The promotion agent flow is:

1. Prompt Guard checks for prompt injection and jailbreaks.
2. PostgreSQL filters out promotions whose parsed end date is before today.
  Promotions without a date remain eligible.
3. pgvector performs semantic search and applies the relevance threshold.
4. `openai/gpt-oss-20b` checks whether the retrieved context is relevant.
5. The answer model responds using only the approved promotion context.

The agent uses PostgreSQL for two purposes:

- `PostgresStore` stores promotion facts and retrieves them semantically through
  pgvector.

Every agent startup resynchronizes `promotion_results.json`. The source filename
is used as the memory key, so an updated record replaces the prior record.

Relevant `.env` options include:

```env
GROQ_API_KEY=your-groq-api-key
GEMINI_API_KEY=your-gemini-api-key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/promotions?sslmode=disable
PROMOTION_RELEVANCE_THRESHOLD=0.30
PROMOTION_ANSWER_MODEL=openai/gpt-oss-120b
USE_GUARD=true
```

Set `USE_GUARD=false` to bypass Prompt Guard. The Prompt Guard model is
downloaded from Hugging Face and may require `HF_TOKEN`.

## LangGraph Studio

The graph configured in `langgraph.json` points to `middleware.py:agent`.
Start the local Studio server with:

```bash
uv run langgraph dev
```

This command launches the local API and opens the LangGraph Studio UI in your
browser. To prevent automatic browser launch and open the printed Studio URL
manually, use:

```bash
uv run langgraph dev --no-browser
```

There is no separate `langgraph ui` command in the current CLI. Use a valid LangSmith key for
tracing, or disable tracing in `.env`:

```env
LANGSMITH_TRACING=false
```

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
