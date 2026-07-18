# Braintrust traced agent demo

A deliberately small terminal agent that demonstrates how Braintrust captures:

- the full terminal session
- each user turn
- automatically instrumented OpenAI calls
- explicit calculator and web-search tool spans

The tools stay intentionally simple:

- **Math.js** handles arithmetic and unit conversions through one free REST endpoint.
- **Tavily** handles current-information searches.

## Setup

Requires Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Export the values from `.env`, or use your preferred environment loader:

```bash
export BRAINTRUST_API_KEY="..."
export OPENAI_API_KEY="..."
export TAVILY_API_KEY="..."
```

Tavily is optional. Arithmetic and unit conversion still work without it.

## Run

```bash
python agent.py
```

To record every chat turn as a separate root trace while grouping all turns by
the same `metadata.session_id`, run:

```bash
python agent_session_traces.py
```

Try:

```text
What is 18 * (7 + 3)?
Convert 72 degrees Fahrenheit to Celsius.
How many kilometers are in 26.2 miles?
What are the latest major announcements from OpenAI?
```

## Expected trace

```text
terminal_chat_session
└── chat_turn
    ├── openai.responses
    ├── calculate or web_search
    └── openai.responses
```

`braintrust.auto_instrument()` captures OpenAI inputs, outputs, latency, token usage, and cost. The `@traced` decorators create readable application and tool spans around those model calls.

The second implementation has no enclosing session span, so each `chat_turn`
starts a new trace. Braintrust's session view groups those traces using their
shared `metadata.session_id`.

## Why the import happens inside `chat()`

Braintrust must initialize and enable auto-instrumentation before the OpenAI client is created. Keeping the OpenAI import next to that startup sequence makes the ordering obvious in a teaching demo.
