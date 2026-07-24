#!/usr/bin/env python3
"""Financial analyst agent that traces to Braintrust, LangSmith, and Langfuse."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from typing import Any

os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")

import braintrust
import langsmith as ls
from braintrust import traced as braintrust_traced
from langfuse import get_client, observe, propagate_attributes
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
BRAINTRUST_PROJECT = os.getenv(
    "BRAINTRUST_PROJECT",
    "Financial Analyst Agent - All Tracing",
)
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", BRAINTRUST_PROJECT)
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


@braintrust_traced(name="get_market_quote", type="tool")
@traceable(name="get_market_quote", run_type="tool", project_name=LANGSMITH_PROJECT)
@observe(name="get_market_quote", as_type="tool")
def get_market_quote(symbol: str) -> str:
    return get_market_quote_impl(symbol)


@braintrust_traced(name="financial_calculator", type="tool")
@traceable(name="financial_calculator", run_type="tool", project_name=LANGSMITH_PROJECT)
@observe(name="financial_calculator", as_type="tool")
def financial_calculator(operation: str, values: dict[str, Any]) -> str:
    return financial_calculator_impl(operation, values)


@braintrust_traced(name="web_search", type="tool")
@traceable(name="web_search", run_type="tool", project_name=LANGSMITH_PROJECT)
@observe(name="web_search", as_type="tool")
def web_search(query: str, max_results: int = 3) -> str:
    return web_search_impl(query, max_results=max_results)


@braintrust_traced(name="expert_financial_analysis", type="tool")
@traceable(
    name="expert_financial_analysis",
    run_type="tool",
    project_name=LANGSMITH_PROJECT,
)
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


@braintrust_traced(
    name="openai.chat.completions.create",
    type="llm",
    notrace_io=True,
)
@traceable(
    name="openai.chat.completions.create",
    run_type="llm",
    project_name=LANGSMITH_PROJECT,
    process_inputs=_call_model_inputs,
    process_outputs=dump_response,
    metadata={"ls_provider": "openai", "ls_model_name": MODEL},
)
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
    ) as langfuse_generation:
        response = client.chat.completions.create(**request)
        metrics = usage_metrics(response)
        response_payload = dump_response(response)
        langfuse_generation.update(
            output=response_payload,
            usage_details={
                "input_tokens": metrics.get("prompt_tokens"),
                "output_tokens": metrics.get("completion_tokens"),
                "total_tokens": metrics.get("tokens"),
            },
            metadata={"provider": "openai"},
        )

    braintrust.current_span().log(
        input=request,
        output=response_payload,
        metadata={"model": MODEL, "provider": "openai"},
        metrics=metrics,
    )
    return response


@braintrust_traced(name="financial_analyst_turn")
@traceable(
    name="financial_analyst_turn",
    run_type="chain",
    project_name=LANGSMITH_PROJECT,
    process_inputs=_chat_turn_inputs,
)
@observe(name="financial_analyst_turn", as_type="span")
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

    answer = "Stopped after too many tool calls."
    braintrust.current_span().log(
        input=user_input,
        output=answer,
        metadata={"model": MODEL, "session_id": session_id},
    )
    return answer


def flush_langsmith() -> None:
    result = ls.Client().flush()
    if inspect.isawaitable(result):
        asyncio.run(result)


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("BRAINTRUST_API_KEY"):
        raise SystemExit("Set BRAINTRUST_API_KEY before running the demo.")
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Set LANGSMITH_API_KEY before running the demo.")
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        raise SystemExit("Set LANGFUSE_PUBLIC_KEY before running the demo.")
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        raise SystemExit("Set LANGFUSE_SECRET_KEY before running the demo.")

    braintrust_logger = braintrust.init_logger(project=BRAINTRUST_PROJECT)
    langfuse = get_client()
    client = OpenAI()
    
    conversation: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    session_id = str(uuid.uuid4())
    user_inputs: list[str] = []
    agent_outputs: list[str] = []

    print(f"All-tracing financial analyst agent using {MODEL}")
    print(f"Braintrust project: {BRAINTRUST_PROJECT}")
    print(f"LangSmith project: {LANGSMITH_PROJECT}")
    print("Langfuse project is selected by LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY.")
    print("Try one of these:")
    for question in EXAMPLE_QUESTIONS:
        print(f"- {question}")
    print("Type \\quit to exit.\n")

    try:
        with braintrust_logger.start_span(
            name="financial_analyst_session",
            metadata={
                "model": MODEL,
                "session_id": session_id,
                "langsmith_project": LANGSMITH_PROJECT,
            },
            tags=["financial-agent", "all-tracing"],
        ) as braintrust_session:
            with ls.trace(
                "financial_analyst_session",
                "chain",
                project_name=LANGSMITH_PROJECT,
                inputs={"session_id": session_id, "model": MODEL},
                tags=["financial-agent", "all-tracing"],
            ) as langsmith_session:
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="financial_analyst_session",
                    input={"session_id": session_id, "model": MODEL},
                ) as langfuse_session:
                    with propagate_attributes(
                        session_id=session_id,
                        metadata={"agent": "financial_analyst"},
                        tags=["financial-agent", "all-tracing"],
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
                            answer = chat_turn(
                                client,
                                conversation,
                                user_input,
                                session_id,
                            )
                            agent_outputs.append(answer)
                            print(f"Agent: {answer}\n")

                    session_output = agent_outputs[-1] if agent_outputs else None
                    session_payload = {
                        "last_response": session_output,
                        "turn_count": len(agent_outputs),
                        "inputs": user_inputs,
                    }
                    langfuse_session.update(output=session_payload)
                    langsmith_session.end(outputs=session_payload)
                    braintrust_session.log(
                        input=user_inputs,
                        output=session_output,
                        metadata={
                            "session_id": session_id,
                            "turn_count": len(agent_outputs),
                        },
                    )
    finally:
        braintrust_logger.flush()
        flush_langsmith()
        langfuse.flush()


if __name__ == "__main__":
    main()
