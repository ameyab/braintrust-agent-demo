#!/usr/bin/env python3
"""Minimal streaming tool-using AI agent with Braintrust tracing."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable

import braintrust
import requests
from braintrust import traced

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROJECT = os.getenv("BRAINTRUST_PROJECT", "demo_ai_agent")
QUIT_COMMANDS = {"\\q", "\\quit", "\\exit"}

SYSTEM_PROMPT = """
You are a concise terminal assistant.

Use the calculate tool for arithmetic and unit conversions.
Use the web_search tool for questions that require current information.
Use tools only when they are needed, and base your answer on their output.
""".strip()

TOOLS = [
    {
        "type": "function",
        "name": "calculate",
        "description": (
            "Evaluate arithmetic or convert units. "
            "Examples: '18 * (7 + 3)' and '10 miles to km'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Math.js-compatible expression.",
                }
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "web_search",
        "description": "Search the web for current information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A concise web search query.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def calculate(expression: str) -> str:
    """Evaluate arithmetic or unit conversion with the free Math.js API."""
    response = requests.get(
        "https://api.mathjs.org/v4/",
        params={"expr": expression},
        timeout=10,
    )
    response.raise_for_status()
    return response.text


def web_search(query: str) -> str:
    """Search Tavily and return a compact set of results."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable because TAVILY_API_KEY is not set."

    response = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": query,
            "search_depth": "basic",
            "max_results": 3,
        },
        timeout=15,
    )
    response.raise_for_status()

    results = response.json().get("results", [])
    compact_results = [
        {
            "title": result.get("title"),
            "url": result.get("url"),
            "content": result.get("content"),
        }
        for result in results
    ]
    return json.dumps(compact_results, ensure_ascii=False)


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "web_search": web_search,
}


@traced(name="chat_turn")
def answer_question(
    client: Any,
    conversation: list[dict[str, Any]],
    question: str,
    session_id: str,
) -> str:
    """Run one agent turn and print model text as streaming deltas arrive."""
    braintrust.current_span().log(metadata={"session_id": session_id})
    conversation.append({"role": "user", "content": question})

    for _ in range(5):
        streamed_text = False

        # Keep the stream context open until all events have been consumed.
        # This lets OpenAI and Braintrust finalize the response and its usage data.
        with client.responses.stream(
            model=MODEL,
            instructions=SYSTEM_PROMPT,
            input=conversation,
            tools=TOOLS,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    print(event.delta, end="", flush=True)
                    streamed_text = True

            response = stream.get_final_response()

        # Streaming helpers may attach parsed_arguments for local use. It is
        # not a valid Responses API input field, so do not send it back.
        conversation.extend(
            item.model_dump(
                exclude_none=True,
                exclude={"parsed_arguments"},
            )
            for item in response.output
        )

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            answer = response.output_text or "The model returned no answer."
            if not streamed_text:
                print(answer, end="", flush=True)
            conversation.append({"role": "assistant", "content": answer})
            return answer

        for call in tool_calls:
            try:
                arguments = json.loads(call.arguments or "{}")
                handler = TOOL_HANDLERS[call.name]
                output = handler(**arguments)
            except Exception as exc:
                output = f"Tool error: {type(exc).__name__}: {exc}"

            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    answer = "Stopped after too many tool calls."
    print(answer, end="", flush=True)
    return answer


def chat() -> None:
    """Start an interactive terminal session."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("BRAINTRUST_API_KEY"):
        raise SystemExit("Set BRAINTRUST_API_KEY before running the demo.")

    # Initialization is unchanged for streaming: instrument OpenAI before
    # importing it or constructing its client.
    logger = braintrust.init_logger(project=PROJECT)
    braintrust.auto_instrument()

    from openai import OpenAI

    client = OpenAI()
    conversation: list[dict[str, Any]] = []

    print(f"Streaming demo agent using {MODEL}")
    print("Try math, unit conversion, or a current-information question.")
    print("Type \\quit to exit.\n")

    session_id = str(uuid.uuid4())
    with logger.start_span(
        name="Chat Session",
        metadata={"model": MODEL, "session_id": session_id},
        tags=["Streaming"],
    ):
        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not question:
                continue
            if question.lower() in QUIT_COMMANDS:
                break

            print("Agent: ", end="", flush=True)
            answer_question(client, conversation, question, session_id)
            print("\n")

    logger.flush()


if __name__ == "__main__":
    chat()
