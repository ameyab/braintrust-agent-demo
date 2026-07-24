#!/usr/bin/env python3
"""Shared financial analyst agent logic with no tracing vendor imports."""

from __future__ import annotations

import csv
import json
import os
from io import StringIO
from typing import Any, Callable

import requests

SYSTEM_PROMPT = """
You are a concise financial analyst assistant.

Help users analyze public companies, securities, portfolios, and financial scenarios.
Always use get_market_quote for current or recently delayed prices, quotes,
market data, or ticker lookups. Map common company names to tickers when clear,
for example Microsoft=MSFT, Nvidia/NVIDIA=NVDA, Apple=AAPL, Amazon=AMZN,
Google/Alphabet=GOOGL, Meta=META, Tesla=TSLA.
Always use financial_calculator for deterministic finance math.
Use web_search for recent news, catalysts, filings, macro events, or anything
that may have changed recently.
Use expert_financial_analysis when the user asks for investment analysis,
portfolio fit, risks, bull/bear cases, valuation framing, or advisor-style
judgment. Keep advice educational and general; do not give personalized
instructions to buy, sell, or hold.
State assumptions clearly and include the relevant numbers you used.
Do not provide personalized investment advice, guarantees, or instructions to buy or sell.
""".strip()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_market_quote",
            "description": (
                "Fetch a current or recently delayed quote for a public security. "
                "US stock symbols can be provided as AAPL, MSFT, NVDA, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Ticker symbol, for example AAPL or MSFT.",
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "financial_calculator",
            "description": (
                "Run finance calculations such as cagr, percentage_change, "
                "compound_value, portfolio_return, and valuation_multiple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "cagr",
                            "percentage_change",
                            "compound_value",
                            "portfolio_return",
                            "valuation_multiple",
                        ],
                    },
                    "values": {
                        "type": "object",
                        "description": (
                            "Inputs for the selected operation. Examples: "
                            "cagr needs beginning_value, ending_value, years; "
                            "percentage_change needs initial_value, final_value; "
                            "compound_value needs principal, annual_rate_percent, years; "
                            "portfolio_return needs weights and returns_percent arrays "
                            "where weights sum to 1.0 or 100; "
                            "valuation_multiple needs numerator and denominator."
                        ),
                        "additionalProperties": True,
                    },
                },
                "required": ["operation", "values"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet for recent market news, company catalysts, "
                "earnings updates, analyst commentary, macro context, or filings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of search results to return, from 1 to 5.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expert_financial_analysis",
            "description": (
                "Produce expert investment-analysis framing: thesis, risks, "
                "valuation considerations, catalysts, portfolio fit, and due "
                "diligence checklist. This is educational, not personalized advice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's investment-analysis question.",
                    },
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Relevant tickers, if any.",
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": [
                            "investment_thesis",
                            "risk_review",
                            "portfolio_fit",
                            "valuation_review",
                            "bull_bear_case",
                            "due_diligence",
                        ],
                    },
                    "time_horizon": {
                        "type": "string",
                        "description": "Time horizon such as short-term, 1 year, or long-term.",
                    },
                    "risk_profile": {
                        "type": "string",
                        "description": "General risk profile such as conservative, balanced, or aggressive.",
                    },
                    "context": {
                        "type": "string",
                        "description": "Extra facts or constraints provided by the user.",
                    },
                },
                "required": ["question", "analysis_type"],
                "additionalProperties": False,
            },
        },
    },
]

ToolHandler = Callable[..., str]

QUOTE_TERMS = (
    "price",
    "quote",
    "trading",
    "stock",
    "share",
    "market data",
    "today",
    "latest",
)
CALCULATOR_TERMS = (
    "cagr",
    "compound",
    "percentage",
    "percent",
    "return",
    "valuation",
    "multiple",
)
WEB_SEARCH_TERMS = (
    "news",
    "recent",
    "latest",
    "catalyst",
    "earnings",
    "filing",
    "macro",
    "tariff",
    "regulation",
    "search",
)
ANALYSIS_TERMS = (
    "analyze",
    "analysis",
    "advisor",
    "advice",
    "invest",
    "investment",
    "buy",
    "sell",
    "hold",
    "portfolio",
    "risk",
    "bull",
    "bear",
    "thesis",
    "valuation",
)
COMMON_TICKERS = {
    "ALPHABET": "GOOGL",
    "ALPHABET INC": "GOOGL",
    "ALPHABET INC.": "GOOGL",
    "AMAZON": "AMZN",
    "AMAZON.COM": "AMZN",
    "AMAZON.COM INC": "AMZN",
    "APPLE": "AAPL",
    "APPLE INC": "AAPL",
    "APPLE INC.": "AAPL",
    "GOOGLE": "GOOGL",
    "META": "META",
    "META PLATFORMS": "META",
    "MICROSOFT": "MSFT",
    "MICROSOFT CORPORATION": "MSFT",
    "MICROSOFT CORP": "MSFT",
    "NVIDIA": "NVDA",
    "NVIDIA CORPORATION": "NVDA",
    "NVIDIA CORP": "NVDA",
    "NVDA": "NVDA",
    "TESLA": "TSLA",
    "TESLA INC": "TSLA",
}
EXAMPLE_QUESTIONS = (
    "What's the NVIDIA stock price today?",
    "Search for recent Microsoft AI news, then summarize the likely investment implications.",
    "Compare MSFT and NVDA latest prices, then calculate the percent difference.",
    "What CAGR turns 25,000 into 40,000 over 6 years?",
    "Act like an expert investment advisor and give me a bull/bear case for Apple over the next year.",
    "If a portfolio is 60% NVDA and 40% MSFT with returns of 18% and 9%, what is the portfolio return?",
    "Analyze whether Tesla looks more like a growth, value, or high-risk momentum position.",
)


def _as_float(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc


def _required_number(values: dict[str, Any], *names: str) -> float:
    for name in names:
        if name in values:
            return _as_float(values[name], name)
    raise ValueError(f"Missing required value: {' or '.join(names)}")


def supports_custom_temperature(model: str) -> bool:
    """GPT-5 chat models currently accept only the default temperature."""
    return not model.startswith("gpt-5")


def chat_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": tool_choice,
    }
    if supports_custom_temperature(model):
        kwargs["temperature"] = 0.2
    return kwargs


def initial_tool_choice(user_input: str) -> str | dict[str, Any]:
    """Force a first tool call for common finance asks that smaller models dodge."""
    lower = user_input.lower()
    if any(term in lower for term in WEB_SEARCH_TERMS) and not any(
        term in lower for term in ("price", "quote")
    ):
        return {"type": "function", "function": {"name": "web_search"}}
    if any(term in lower for term in QUOTE_TERMS):
        return {"type": "function", "function": {"name": "get_market_quote"}}
    if any(term in lower for term in CALCULATOR_TERMS):
        return {"type": "function", "function": {"name": "financial_calculator"}}
    if any(term in lower for term in ANALYSIS_TERMS):
        return {"type": "function", "function": {"name": "expert_financial_analysis"}}
    return "auto"


def _optional_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, "", "N/D"):
        return None
    return _as_float(value, field)


def _normalized_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper().lstrip("$")
    if not normalized:
        raise ValueError("symbol is required")
    return COMMON_TICKERS.get(normalized, normalized)


def _stooq_symbol(symbol: str) -> str:
    normalized = _normalized_symbol(symbol)
    if "." in normalized or normalized.startswith("^"):
        return normalized.lower()
    return f"{normalized.lower()}.us"


def _quote_from_yahoo(symbol: str) -> dict[str, Any]:
    response = requests.get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={"interval": "1d", "range": "1d"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()

    result = (data.get("chart", {}).get("result") or [None])[0]
    if not result:
        error = data.get("chart", {}).get("error")
        raise ValueError(f"Yahoo Finance returned no quote data: {error}")

    meta = result.get("meta") or {}
    indicators = result.get("indicators") or {}
    quote_rows = indicators.get("quote") or [{}]
    quote = quote_rows[0] if quote_rows else {}
    timestamps = result.get("timestamp") or []

    price = meta.get("regularMarketPrice") or meta.get("previousClose")
    if price is None:
        closes = [value for value in quote.get("close", []) if value is not None]
        if closes:
            price = closes[-1]
    if price is None:
        raise ValueError("Yahoo Finance quote did not include a price")

    return {
        "symbol": meta.get("symbol") or symbol,
        "price": _as_float(price, "price"),
        "open": _last_number(quote.get("open")),
        "high": _last_number(quote.get("high")),
        "low": _last_number(quote.get("low")),
        "volume": _last_number(quote.get("volume")),
        "currency": meta.get("currency"),
        "exchange": meta.get("fullExchangeName") or meta.get("exchangeName"),
        "timestamp": timestamps[-1] if timestamps else meta.get("regularMarketTime"),
        "provider": "yahoo_finance_chart",
        "note": "Public quote data may be delayed.",
    }


def _quote_from_finnhub(symbol: str) -> dict[str, Any]:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise ValueError("FINNHUB_API_KEY is not set")

    response = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={"symbol": symbol, "token": api_key},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    price = data.get("c")
    if not price:
        raise ValueError(f"Finnhub returned no current price for {symbol}")

    return {
        "symbol": symbol,
        "price": _as_float(price, "c"),
        "open": data.get("o"),
        "high": data.get("h"),
        "low": data.get("l"),
        "previous_close": data.get("pc"),
        "change": data.get("d"),
        "change_percent": data.get("dp"),
        "currency": "USD",
        "timestamp": data.get("t"),
        "provider": "finnhub",
        "note": "Freshness depends on your Finnhub plan.",
    }


def _quote_from_alpha_vantage(symbol: str) -> dict[str, Any]:
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is not set")

    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": api_key,
            "entitlement": "delayed",
        },
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    quote = data.get("Global Quote") or {}
    price = quote.get("05. price")
    if not price:
        message = data.get("Note") or data.get("Information") or data.get("Error Message")
        raise ValueError(f"Alpha Vantage returned no quote for {symbol}: {message}")

    return {
        "symbol": quote.get("01. symbol") or symbol,
        "price": _as_float(price, "05. price"),
        "open": _as_float(quote.get("02. open"), "02. open"),
        "high": _as_float(quote.get("03. high"), "03. high"),
        "low": _as_float(quote.get("04. low"), "04. low"),
        "volume": _as_float(quote.get("06. volume"), "06. volume"),
        "latest_trading_day": quote.get("07. latest trading day"),
        "previous_close": _as_float(quote.get("08. previous close"), "08. previous close"),
        "change": _as_float(quote.get("09. change"), "09. change"),
        "change_percent": quote.get("10. change percent"),
        "currency": "USD",
        "provider": "alpha_vantage",
        "note": "Freshness depends on your Alpha Vantage entitlement.",
    }


def _last_number(values: Any) -> float | None:
    if not isinstance(values, list):
        return None
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return _as_float(numbers[-1], "value")


def get_market_quote_impl(symbol: str) -> str:
    """Fetch quote data through a provider chain with no-key fallback."""
    normalized_symbol = _normalized_symbol(symbol)
    provider_errors: list[str] = []

    providers = []
    if os.getenv("FINNHUB_API_KEY"):
        providers.append(("finnhub", _quote_from_finnhub))
    if os.getenv("ALPHA_VANTAGE_API_KEY"):
        providers.append(("alpha_vantage", _quote_from_alpha_vantage))
    providers.extend(
        [
            ("yahoo_finance_chart", _quote_from_yahoo),
            ("stooq", _quote_from_stooq),
        ]
    )

    for provider_name, provider in providers:
        try:
            payload = provider(normalized_symbol)
            payload["requested_symbol"] = symbol
            return json.dumps(payload)
        except Exception as exc:
            provider_errors.append(f"{provider_name}: {type(exc).__name__}: {exc}")

    return json.dumps(
        {
            "symbol": normalized_symbol,
            "requested_symbol": symbol,
            "error": "All quote providers failed.",
            "provider_errors": provider_errors,
        }
    )


def web_search_impl(query: str, max_results: int = 3) -> str:
    """Search Tavily and return compact internet results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return json.dumps(
            {
                "query": query,
                "error": "TAVILY_API_KEY is not set, so web search is unavailable.",
            }
        )

    bounded_results = max(1, min(int(max_results or 3), 5))
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": bounded_results,
            },
            timeout=15,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        return json.dumps(
            {
                "query": query,
                "provider": "tavily",
                "error": f"{type(exc).__name__}: {exc}",
            },
            ensure_ascii=False,
        )

    compact_results = [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
            "score": result.get("score"),
        }
        for result in results
    ]
    return json.dumps(
        {
            "query": query,
            "provider": "tavily",
            "results": compact_results,
        },
        ensure_ascii=False,
    )


def expert_financial_analysis_impl(
    question: str,
    analysis_type: str,
    symbols: list[str] | None = None,
    time_horizon: str | None = None,
    risk_profile: str | None = None,
    context: str | None = None,
) -> str:
    """Return an expert-style investment analysis framework."""
    normalized_symbols = [_normalized_symbol(symbol) for symbol in (symbols or [])]
    quote_snapshots = []
    for symbol in normalized_symbols[:3]:
        try:
            quote_snapshots.append(json.loads(get_market_quote_impl(symbol)))
        except Exception as exc:
            quote_snapshots.append(
                {
                    "symbol": symbol,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    analysis = {
        "analysis_type": analysis_type,
        "question": question,
        "symbols": normalized_symbols,
        "time_horizon": time_horizon or "not specified",
        "risk_profile": risk_profile or "not specified",
        "context": context or "",
        "quote_snapshots": quote_snapshots,
        "investment_framework": {
            "thesis": [
                "Identify the company's durable revenue drivers and whether they are accelerating or maturing.",
                "Compare current price action with business fundamentals, not just recent headlines.",
                "Separate cyclical factors from structural advantages such as scale, distribution, data, or switching costs.",
            ],
            "valuation": [
                "Compare valuation multiples to growth, margin quality, free cash flow conversion, and balance-sheet risk.",
                "Use scenario analysis rather than a single point estimate.",
                "Check whether the market is pricing perfection, normalization, or distress.",
            ],
            "risks": [
                "Monitor execution risk, competitive pressure, regulation, customer concentration, and macro sensitivity.",
                "For high-growth names, watch multiple compression and revenue deceleration.",
                "For mature names, watch capital allocation discipline and margin durability.",
            ],
            "catalysts": [
                "Upcoming earnings, guidance revisions, product cycles, regulatory decisions, and rate expectations can change the setup.",
                "Recent news should be verified with web_search before treating it as part of the thesis.",
            ],
            "portfolio_fit": [
                "Size exposure based on volatility, correlation with existing holdings, liquidity needs, and drawdown tolerance.",
                "Avoid treating a strong company as automatically attractive at any price.",
            ],
        },
        "advisor_style_takeaway": (
            "I can provide an expert analytical framework and highlight tradeoffs, "
            "but this is not personalized financial advice. A final decision should "
            "depend on your full financial situation, constraints, and risk tolerance."
        ),
        "suggested_next_steps": [
            "Use get_market_quote for the latest available price.",
            "Use web_search for recent catalysts or company-specific news.",
            "Use financial_calculator for explicit return, CAGR, compounding, or valuation-multiple math.",
        ],
    }
    return json.dumps(analysis, ensure_ascii=False)


def _quote_from_stooq(symbol: str) -> dict[str, Any]:
    stooq_symbol = _stooq_symbol(symbol)
    response = requests.get(
        "https://stooq.com/q/l/",
        params={"s": stooq_symbol, "f": "sd2t2ohlcv", "h": "", "e": "csv"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    response.raise_for_status()

    rows = list(csv.DictReader(StringIO(response.text)))
    if not rows:
        raise ValueError("No quote rows returned by provider.")

    row = rows[0]
    close = row.get("Close")
    if not close or close == "N/D":
        raise ValueError("Quote unavailable from provider.")

    return {
        "symbol": _normalized_symbol(symbol),
        "provider_symbol": stooq_symbol,
        "price": _as_float(close, "Close"),
        "open": _optional_float(row, "Open"),
        "high": _optional_float(row, "High"),
        "low": _optional_float(row, "Low"),
        "volume": _optional_float(row, "Volume"),
        "currency": "USD" if stooq_symbol.endswith(".us") else "unknown",
        "date": row.get("Date"),
        "time": row.get("Time"),
        "provider": "stooq",
        "note": "Public quote data may be delayed.",
    }


def financial_calculator_impl(operation: str, values: dict[str, Any]) -> str:
    """Run small deterministic finance calculations."""
    op = operation.strip().lower()
    values = values or {}

    if op == "cagr":
        beginning = _required_number(values, "beginning_value", "initial_value")
        ending = _required_number(values, "ending_value", "final_value")
        years = _required_number(values, "years")
        if beginning <= 0 or ending <= 0 or years <= 0:
            raise ValueError("beginning_value, ending_value, and years must be positive")
        result = ((ending / beginning) ** (1 / years) - 1) * 100
        payload = {
            "operation": op,
            "result_percent": result,
            "formula": "(ending_value / beginning_value) ** (1 / years) - 1",
        }
    elif op == "percentage_change":
        initial = _required_number(values, "initial_value", "beginning_value")
        final = _required_number(values, "final_value", "ending_value")
        if initial == 0:
            raise ValueError("initial_value cannot be zero")
        result = ((final - initial) / initial) * 100
        payload = {
            "operation": op,
            "result_percent": result,
            "formula": "(final_value - initial_value) / initial_value",
        }
    elif op == "compound_value":
        principal = _required_number(values, "principal", "initial_value")
        annual_rate = _required_number(values, "annual_rate_percent", "rate_percent") / 100
        years = _required_number(values, "years")
        if years < 0:
            raise ValueError("years cannot be negative")
        result = principal * ((1 + annual_rate) ** years)
        payload = {
            "operation": op,
            "result_value": result,
            "formula": "principal * (1 + annual_rate) ** years",
        }
    elif op == "portfolio_return":
        weights = values.get("weights")
        returns = values.get("returns_percent")
        if not isinstance(weights, list) or not isinstance(returns, list):
            raise ValueError("weights and returns_percent must be arrays")
        if len(weights) != len(returns) or not weights:
            raise ValueError("weights and returns_percent must have the same non-zero length")
        weight_values = [_as_float(weight, "weight") for weight in weights]
        return_values = [_as_float(ret, "return_percent") for ret in returns]
        weight_sum = sum(weight_values)
        if weight_sum > 1.5:
            weight_values = [weight / 100 for weight in weight_values]
            weight_sum = sum(weight_values)
        result = sum(weight * ret for weight, ret in zip(weight_values, return_values))
        payload = {
            "operation": op,
            "result_percent": result,
            "formula": "sum(weight * return_percent)",
            "weight_sum": weight_sum,
        }
    elif op == "valuation_multiple":
        numerator = _required_number(
            values,
            "numerator",
            "enterprise_value",
            "market_cap",
            "price",
        )
        denominator = _required_number(
            values,
            "denominator",
            "ebitda",
            "revenue",
            "earnings",
        )
        if denominator == 0:
            raise ValueError("denominator cannot be zero")
        result = numerator / denominator
        payload = {
            "operation": op,
            "result_multiple": result,
            "formula": "numerator / denominator",
        }
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    payload["inputs"] = values
    return json.dumps(payload)


def usage_metrics(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    metrics: dict[str, int] = {}
    for attr, key in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "tokens"),
    ):
        value = getattr(usage, attr, None)
        if value is not None:
            metrics[key] = value
    return metrics


def dump_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)
    return {"response": str(response)}


def tool_call_to_message(tool_call: Any, output: str) -> dict[str, str]:
    return {
        "role": "tool",
        "tool_call_id": tool_call.id,
        "name": tool_call.function.name,
        "content": output,
    }


def run_tool_call(tool_call: Any, handlers: dict[str, ToolHandler]) -> str:
    name = tool_call.function.name
    try:
        arguments = json.loads(tool_call.function.arguments or "{}")
        handler = handlers[name]
        return handler(**arguments)
    except Exception as exc:
        return json.dumps(
            {
                "tool": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
