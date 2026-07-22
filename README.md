# Braintrust traced agent demo

A deliberately small terminal agent that demonstrates how Braintrust captures:

- the full terminal session
- each user turn
- manually traced OpenAI calls
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
python simple_agent.py
```

To record every chat turn as a separate root trace while grouping all turns by
the same `metadata.session_id`, set `GROUP_AS_CONVERSATION = False` in
`simple_agent.py`, or run:

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
Chat Session
└── chat_turn
    ├── openai.responses.create
    ├── calculate or web_search
    └── openai.responses.create
```

The `@traced` decorators create readable application, LLM, and tool spans. OpenAI
inputs, outputs, latency, and related metadata are logged explicitly in
`call_model()` rather than via auto-instrumentation.

When `GROUP_AS_CONVERSATION` is `False`, there is no enclosing session span, so
each `chat_turn` starts a new trace. Braintrust's session view groups those
traces using their shared `metadata.session_id`.

## Why the import happens inside `main()`

Braintrust must initialize before the OpenAI client is created. Keeping the
OpenAI import next to that startup sequence makes the ordering obvious in a
teaching demo.

## Remote eval

`simple_agent_remote_eval.py` runs `simple_agent.py` against a Braintrust dataset
and uses a separate LLM as a simulated user to continue each conversation.
The remote eval exposes parameters for the agent model, simulated-user model,
judge model, maximum turns, sampling temperature, agent system prompt, and the
descriptions for the `calculate` and `web_search` tools.

By default, it reads the `Simple Agent Conversations` dataset in the `Simple Agent`
project. Override these values if your dataset has a different name:

```bash
export BRAINTRUST_PROJECT="Simple Agent"
export BRAINTRUST_DATASET="Simple Agent Conversations"
```

Dataset rows can be plain strings or objects. For object rows, the eval looks for
fields such as `goal`, `user_goal`, `scenario`, `initial_message`, `question`,
`prompt`, `context`, and `success_criteria`. The row `expected` value is used as
the judge criteria when present.

Run once from the CLI:

```bash
bt eval simple_agent_remote_eval.py --language python
```

Expose it as a remote eval source for the Braintrust playground:

```bash
bt eval simple_agent_remote_eval.py --language python --dev
```

Then add `http://localhost:8300` in the project settings under Remote evals.
