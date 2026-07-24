#!/usr/bin/env python3
"""Financial analyst agent that traces only to Langfuse."""

from __future__ import annotations

import os
import uuid
from typing import Any

from langfuse import get_client, observe, propagate_attributes
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
QUIT_COMMANDS = {"\\q", "\\quit", "\\exit", "quit", "exit"}


@observe(name="get_market_quote", as_type="tool")
def get_market_quote(symbol: str) -> str:
    return get_market_quote_impl(symbol)


@observe(name="financial_calculator", as_type="tool")
def financial_calculator(operation: str, values: dict[str, Any]) -> str:
    return financial_calculator_impl(operation, values)


@observe(name="web_search", as_type="tool")
def web_search(query: str, max_results: int = 3) -> str:
    return web_search_impl(query, max_results=max_results)


@observe(name="expert_financial_analysis", as_type="tool")
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


def call_model(
    client: OpenAI,
    messages: list[dict[str, Any]],
    tool_choice: str | dict[str, Any] = "auto",
) -> Any:
    request = chat_completion_kwargs(MODEL, messages, tool_choice=tool_choice)
    with get_client().start_as_current_observation(
        as_type="generation",
        name="openai.chat.completions.create",
        model=MODEL,
        input=request,
    ) as generation:
        response = client.chat.completions.create(**request)
        metrics = usage_metrics(response)
        generation.update(
            output=dump_response(response),
            usage_details={
                "input_tokens": metrics.get("prompt_tokens"),
                "output_tokens": metrics.get("completion_tokens"),
                "total_tokens": metrics.get("tokens"),
            },
            metadata={"provider": "openai"},
        )
        return response


@observe(name="financial_analyst_turn", as_type="span")
def chat_turn(
    client: OpenAI,
    conversation: list[dict[str, Any]],
    user_input: str,
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


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY before running the demo.")
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        raise SystemExit("Set LANGFUSE_SECRET_KEY before running the demo.")

    langfuse = get_client()
    client = OpenAI()
    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_id = str(uuid.uuid4())
    user_inputs: list[str] = []
    agent_outputs: list[str] = []

    print(f"Langfuse-only financial analyst agent using {MODEL}")
    print("Langfuse project is selected by LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY.")
    print("Try one of these:")
    for question in EXAMPLE_QUESTIONS:
        print(f"- {question}")
    print("Type \\quit to exit.\n")

    try:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="financial_analyst_session",
            input={"session_id": session_id, "model": MODEL},
        ) as session_span:
            with propagate_attributes(
                session_id=session_id,
                metadata={"agent": "financial_analyst"},
                tags=["financial-agent", "langfuse-only"],
                trace_name="financial_analyst_session",
            ):
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
                    answer = chat_turn(client, conversation, user_input)
                    agent_outputs.append(answer)
                    print(f"Agent: {answer}\n")

            session_span.update(
                output={
                    "last_response": agent_outputs[-1] if agent_outputs else None,
                    "turn_count": len(agent_outputs),
                    "inputs": user_inputs,
                }
            )
    finally:
        langfuse.flush()


if __name__ == "__main__":
    main()
