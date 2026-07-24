#!/usr/bin/env python3
"""Financial analyst agent that traces only to Braintrust."""

from __future__ import annotations

import os
import uuid
from typing import Any

import braintrust
from braintrust import traced
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
        usage_metrics,
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
        usage_metrics,
        web_search_impl,
    )

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
PROJECT = os.getenv("BRAINTRUST_PROJECT", "Financial Analyst Agent - Braintrust")
QUIT_COMMANDS = {"\\q", "\\quit", "\\exit", "quit", "exit"}


@traced(name="get_market_quote", type="tool")
def get_market_quote(symbol: str) -> str:
    return get_market_quote_impl(symbol)


@traced(name="financial_calculator", type="tool")
def financial_calculator(operation: str, values: dict[str, Any]) -> str:
    return financial_calculator_impl(operation, values)


@traced(name="web_search", type="tool")
def web_search(query: str, max_results: int = 3) -> str:
    return web_search_impl(query, max_results=max_results)


@traced(name="expert_financial_analysis", type="tool")
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


@traced(name="openai.chat.completions.create", type="llm", notrace_io=True)
def call_model(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    request = chat_completion_kwargs(MODEL, messages, tool_choice=tool_choice)
    response = client.chat.completions.create(**request)
    braintrust.current_span().log(
        input=request,
        output=dump_response(response),
        metadata={"model": MODEL, "provider": "openai"},
        metrics=usage_metrics(response),
    )
    return response


@traced(name="financial_analyst_turn")
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
            answer = assistant_message.content or "The model returned no answer."
            braintrust.current_span().log(
                input=user_input,
                output=answer,
                metadata={"model": MODEL, "session_id": session_id},
            )
            return answer

        for tool_call in tool_calls:
            output = run_tool_call(tool_call, TOOL_HANDLERS)
            conversation.append(tool_call_to_message(tool_call, output))

    return "Stopped after too many tool calls."


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("BRAINTRUST_API_KEY"):
        raise SystemExit("Set BRAINTRUST_API_KEY before running the demo.")

    logger = braintrust.init_logger(project=PROJECT)
    client = OpenAI()
    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_id = str(uuid.uuid4())
    user_inputs: list[str] = []
    agent_outputs: list[str] = []

    print(f"Braintrust-only financial analyst agent using {MODEL}")
    print(f"Braintrust project: {PROJECT}")
    print("Try one of these:")
    for question in EXAMPLE_QUESTIONS:
        print(f"- {question}")
    print("Type \\quit to exit.\n")

    try:
        with logger.start_span(
            name="financial_analyst_session",
            metadata={"model": MODEL, "session_id": session_id},
            tags=["financial-agent", "braintrust-only"],
        ) as session_span:
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

            session_span.log(
                input=user_inputs,
                output=agent_outputs[-1] if agent_outputs else None,
                metadata={"session_id": session_id, "turn_count": len(agent_outputs)},
            )
    finally:
        logger.flush()


if __name__ == "__main__":
    main()
