# Braintrust Agent Demo

Small Python examples for tracing tool-using agents with Braintrust and for
comparing Braintrust instrumentation with LangSmith and Langfuse.

## Folders

- `simple_agent_examples/`: the main Braintrust traced terminal agent, session
  grouping variant, remote eval, and deploy helpers.
- `concepts/`: smaller concept scripts that demonstrate tracing variants such
  as manual LLM spans and streaming.
- `migration_examples/`: side-by-side migration examples for Braintrust,
  LangSmith, and Langfuse.
- `support_chatbot/`: minimal support chatbot using Braintrust OpenAI wrapping.

## Setup

Use a single environment from the repo root for all examples:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Export the values from `.env`, or use your preferred environment loader:

```bash
export OPENAI_API_KEY="..."
export BRAINTRUST_API_KEY="..."
export TAVILY_API_KEY="..."
export LANGSMITH_API_KEY="..."
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
```

`TAVILY_API_KEY` is optional for examples that include web search. Without it,
the web-search tool returns a clear unavailable message. LangSmith and Langfuse
keys are only needed for their migration examples.

## Quick Run

```bash
python simple_agent_examples/simple_agent.py
python support_chatbot/support_chatbot.py
python migration_examples/braintrust_financial_agent.py
python migration_examples/langsmith_financial_agent.py
python migration_examples/langfuse_financial_agent.py
```

See each folder README for the scripts in that area and their expected traces.
