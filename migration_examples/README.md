# Migration Examples

Examples for comparing tracing backends on similar agent flows.

## Support Chatbot Migration

`langsmith_migration.py` traces one support chatbot to both Braintrust and
LangSmith. It is useful for comparing trace shape while migrating.

```bash
python migration_examples/langsmith_migration.py
```

Required:

```bash
export OPENAI_API_KEY="..."
export BRAINTRUST_API_KEY="..."
export LANGSMITH_API_KEY="..."
```

## Financial Analyst Agents

These scripts implement the same financial analyst agent with four tools and
different tracing backends:

- `braintrust_financial_agent.py`: Braintrust only.
- `langsmith_financial_agent.py`: LangSmith only.
- `langfuse_financial_agent.py`: Langfuse only.
- `all_tracing_financial_agent.py`: Braintrust, LangSmith, and Langfuse in the
  same run.

The tools are:

- `get_market_quote`: fetches public quote data. It uses `FINNHUB_API_KEY` or
  `ALPHA_VANTAGE_API_KEY` when configured, then falls back to Yahoo Finance's
  public chart endpoint, then Stooq.
- `financial_calculator`: supports `cagr`, `percentage_change`,
  `compound_value`, `portfolio_return`, and `valuation_multiple`.
- `web_search`: performs internet search through Tavily for recent news,
  catalysts, macro context, filings, and analyst commentary.
- `expert_financial_analysis`: produces expert investment-analysis framing,
  including thesis, risks, valuation considerations, catalysts, portfolio fit,
  and diligence checklist. It is educational and does not provide personalized
  buy/sell instructions.

Example questions:

```text
What's the NVIDIA stock price today?
Search for recent Microsoft AI news, then summarize the likely investment implications.
Compare MSFT and NVDA latest prices, then calculate the percent difference.
What CAGR turns 25,000 into 40,000 over 6 years?
Act like an expert investment advisor and give me a bull/bear case for Apple over the next year.
If a portfolio is 60% NVDA and 40% MSFT with returns of 18% and 9%, what is the portfolio return?
Analyze whether Tesla looks more like a growth, value, or high-risk momentum position.
```

Run from the repo root:

```bash
python migration_examples/braintrust_financial_agent.py
python migration_examples/langsmith_financial_agent.py
python migration_examples/langfuse_financial_agent.py
python migration_examples/all_tracing_financial_agent.py
```

Backend-specific environment:

```bash
export BRAINTRUST_API_KEY="..."
export LANGSMITH_API_KEY="..."
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export TAVILY_API_KEY="..."
```

Optional:

```bash
export OPENAI_MODEL="gpt-5-nano"
export FINNHUB_API_KEY="..."
export ALPHA_VANTAGE_API_KEY="..."
export BRAINTRUST_PROJECT="Financial Analyst Agent - Braintrust"
export LANGSMITH_PROJECT="Financial Analyst Agent - LangSmith"
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

Install dependencies from the repo root:

```bash
pip install -r requirements.txt
```
