# Support Chatbot

A minimal e-commerce support chatbot that uses Braintrust's OpenAI wrapper.

The script demonstrates:

- wrapping an OpenAI client with `braintrust.wrap_openai`
- tracing each model call with `@traced`
- optionally grouping the full conversation under one Braintrust span

## Run

From the repo root:

```bash
python support_chatbot/support_chatbot.py
```

Required:

```bash
export OPENAI_API_KEY="..."
export BRAINTRUST_API_KEY="..."
```

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```
