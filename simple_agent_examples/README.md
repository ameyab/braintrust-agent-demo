# Simple Agent Examples

The main Braintrust traced terminal agent examples. The agent has two tools:

- `calculate`: arithmetic and unit conversions through Math.js.
- `web_search`: current-information search through Tavily.

## Run The Agent

From the repo root:

```bash
python simple_agent_examples/simple_agent.py
```

To record every chat turn as a separate root trace while grouping those traces
by shared `metadata.session_id`, run:

```bash
python simple_agent_examples/agent_session_traces.py
```

Try:

```text
What is 18 * (7 + 3)?
Convert 72 degrees Fahrenheit to Celsius.
How many kilometers are in 26.2 miles?
What are the latest major announcements from OpenAI?
```

## Expected Trace

```text
Chat Session
└── chat_turn
    ├── openai.responses.create
    ├── calculate or web_search
    └── openai.responses.create
```

## Remote Eval

`simple_agent_remote_eval.py` runs the agent against a Braintrust dataset and
uses a separate LLM as a simulated user.

```bash
bt eval simple_agent_examples/simple_agent_remote_eval.py --language python
```

Expose it as a remote eval source for the Braintrust playground:

```bash
bt eval simple_agent_examples/simple_agent_remote_eval.py --language python --dev
```

By default, the eval reads `Popular Usecases Test Dataset` from the configured
Braintrust project. Override as needed:

```bash
export BRAINTRUST_PROJECT="Simple Agent"
export BRAINTRUST_DATASET="Simple Agent Conversations"
```

## Deploy Helpers

Push the demo prompt:

```bash
bt functions push simple_agent_examples/deploy_prompt.py --runner .venv/bin/python \
  --if-exists replace --yes --environment production
```

Push the tool definitions:

```bash
bt functions push simple_agent_examples/deploy_tools.py --runner .venv/bin/python \
  --requirements requirements.txt --if-exists replace --yes
```

## Environment

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Required:

```bash
export OPENAI_API_KEY="..."
export BRAINTRUST_API_KEY="..."
```

Optional:

```bash
export TAVILY_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export BRAINTRUST_PROJECT="Simple Agent"
```
