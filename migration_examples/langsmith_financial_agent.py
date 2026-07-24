#!/usr/bin/env python3
"""Financial analyst agent that traces only to LangSmith."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from typing import Any

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")

import langsmith as ls
from langsmith import traceable
from openai import OpenAI

try:
    from migration_examples.financial_agent_common import (
        EXAMPLE_QUESTIONS,
        SYSTEM_PROMPT,
        TOOLS,
        chat_completion_kwargs,
        dump_response,
        expert_financial_analysis_impl,
        financial_calculator_impl,
        get_market_quote_impl,
        initial_tool_choice,
        run_tool_call,
        tool_call_to_message,
        web_search_impl,
    )
except ModuleNotFoundError:
    from financial_agent_common import (
        EXAMPLE_QUESTIONS,
        SYSTEM_PROMPT,
        TOOLS,
        chat_completion_kwargs,
        dump_response,
        expert_financial_analysis_impl,
        financial_calculator_impl,
        get_market_quote_impl,
        initial_tool_choice,
        run_tool_call,
        tool_call_to_message,
        web_search_impl,
    )

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
PROJECT = os.getenv("LANGSMITH_PROJECT", "Financial Analyst Agent - LangSmith")
QUIT_COMMANDS = {"\\q", "\\quit", "\\exit", "quit", "exit"}


def _call_model_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": inputs.get("messages"),
        "tools": TOOLS,
        "tool_choice": inputs.get("tool_choice"),
        "model": MODEL,
    }


def _chat_turn_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    conversation = inputs.get("conversation") or []
    return {
        "user_input": inputs.get("user_input"),
        "session_id": inputs.get("session_id"),
        "conversation_length": len(conversation),
        "model": MODEL,
    }


@traceable(name="get_market_quote", run_type="tool", project_name=PROJECT)
def get_market_quote(symbol: str) -> str:
    return get_market_quote_impl(symbol)


@traceable(name="financial_calculator", run_type="tool", project_name=PROJECT)
def financial_calculator(operation: str, values: dict[str, Any]) -> str:
    return financial_calculator_impl(operation, values)


@traceable(name="web_search", run_type="tool", project_name=PROJECT)
def web_search(query: str, max_results: int = 3) -> str:
    return web_search_impl(query, max_results=max_results)


@traceable(name="expert_financial_analysis", run_type="tool", project_name=PROJECT)
def expert_financial_analysis(
    question: str,
    analysis_type: str,
    symbols: list[str] | None = None,
    time_horizon: str | None = None,
    risk_profile: str | None = None,
    context: str | None = None,
) -> str:
    return expert_financial_analysis_impl(
        question=question,
        analysis_type=analysis_type,
        symbols=symbols,
        time_horizon=time_horizon,
        risk_profile=risk_profile,
        context=context,
    )


TOOL_HANDLERS = {
    "get_market_quote": get_market_quote,
    "financial_calculator": financial_calculator,
    "web_search": web_search,
    "expert_financial_analysis": expert_financial_analysis,
}


@traceable(
    name="openai.chat.completions.create",
    run_type="llm",
    project_name=PROJECT,
    process_inputs=_call_model_inputs,
    process_outputs=dump_response,
    metadata={"ls_provider": "openai", "ls_model_name": MODEL},
)
def call_model(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    return client.chat.completions.create(
        **chat_completion_kwargs(MODEL, messages, tool_choice=tool_choice)
    )


@traceable(
    name="financial_analyst_turn",
    run_type="chain",
    project_name=PROJECT,
    process_inputs=_chat_turn_inputs,
)
def chat_turn(
    client: OpenAI,
    conversation: list[dict[str, Any]],
    user_input: str,
    session_id: str,
) -> str:
    conversation.append({"role": "user", "content": user_input})
    first_model_call = True

    for _ in range(5):
        response = call_model(
            client,
            conversation,
            tool_choice=initial_tool_choice(user_input) if first_model_call else "auto",
        )
        first_model_call = False
        assistant_message = response.choices[0].message
        conversation.append(assistant_message.model_dump(exclude_none=True))

        tool_calls = assistant_message.tool_calls or []
        if not tool_calls:
            return assistant_message.content or "The model returned no answer."

        for tool_call in tool_calls:
            output = run_tool_call(tool_call, TOOL_HANDLERS)
            conversation.append(tool_call_to_message(tool_call, output))

    return "Stopped after too many tool calls."


def flush_langsmith() -> None:
    result = ls.Client().flush()
    if inspect.isawaitable(result):
        asyncio.run(result)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before running the demo.")

    client = OpenAI()
    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_id = str(uuid.uuid4())
    user_inputs: list[str] = []
    agent_outputs: list[str] = []

    print(f"LangSmith-only financial analyst agent using {MODEL}")
    print(f"LangSmith project: {PROJECT}")
    print("Try one of these:")
    for question in EXAMPLE_QUESTIONS:
        print(f"- {question}")
    print("Type \\quit to exit.\n")

    try:
        with ls.trace(
            "financial_analyst_session",
            "chain",
            project_name=PROJECT,
            inputs={"session_id": session_id, "model": MODEL},
            tags=["financial-agent", "langsmith-only"],
        ) as session:
            while True:
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue
                if user_input.lower() in QUIT_COMMANDS:
                    break

                user_inputs.append(user_input)
                answer = chat_turn(client, conversation, user_input, session_id)
                agent_outputs.append(answer)
                print(f"Agent: {answer}\n")

            session.end(
                outputs={
                    "last_response": agent_outputs[-1] if agent_outputs else None,
                    "turn_count": len(agent_outputs),
                    "inputs": user_inputs,
                }
            )
    finally:
        flush_langsmith()


if __name__ == "__main__":
    main()
