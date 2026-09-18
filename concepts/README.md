# Concepts

Focused Braintrust tracing examples for a calculator/search terminal agent.

## Scripts

- `agent.py`: minimal tool-using agent with Braintrust spans around turns and
  tools.
- `agent_manual_tracing.py`: explicit LLM span logging for OpenAI Responses API
  calls, including token metrics and tool metadata.
- `agent_streaming.py`: streaming variant of the tool-using agent.
- `agent_to_two_projects.py`: placeholder for experimenting with routing traces
  to multiple Braintrust projects.

## Run

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```

Then run a concept script from the repo root:

```bash
python concepts/agent.py
python concepts/agent_manual_tracing.py
python concepts/agent_streaming.py
```

Required environment:

```bash
export OPENAI_API_KEY="..."
export BRAINTRUST_API_KEY="..."
```

Optional:

```bash
export TAVILY_API_KEY="..."
export OPENAI_MODEL="gpt-4o-mini"
export BRAINTRUST_PROJECT="demo_ai_agent"
```
