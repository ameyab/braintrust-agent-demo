#!/usr/bin/env python3
"""Tool-using AI agent with fully manual OpenAI and tool tracing."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Callable

import braintrust
import requests
from braintrust import traced

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GROUP_AS_CONVERSATION = True
if GROUP_AS_CONVERSATION:
    PROJECT = os.getenv("BRAINTRUST_PROJECT", "Simple Agent")
else:
    PROJECT = os.getenv("BRAINTRUST_PROJECT", "Simple Agent - Turn")
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


@traced(name="calculate", type="tool")
def calculate(expression: str) -> str:
    """Evaluate arithmetic or unit conversion with the free Math.js API."""
    response = requests.get(
        "https://api.mathjs.org/v4/",
        params={"expr": expression},
        timeout=10,
    )
    response.raise_for_status()
    return response.text


@traced(name="web_search", type="tool")
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

    metadata = [{"title": result["title"], "url": result["url"]} for result in compact_results]
    braintrust.current_span().log(
        metadata={"search_results": metadata},
    )
    return json.dumps(compact_results, ensure_ascii=False)


TOOL_HANDLERS: dict[str, Callable[..., str]] = {
    "calculate": calculate,
    "web_search": web_search,
}


@traced(name="openai.responses.create", type="llm", notrace_io=True)
def call_model(
    client: Any,
    conversation: list[dict[str, Any]],
    model: str = MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    tools: list[dict[str, Any]] = TOOLS,
) -> Any:
    """Call OpenAI and log the LLM span in Braintrust's standard format."""
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=conversation,
        tools=tools,
    )

    output = [
        item.model_dump(exclude_none=True)
        for item in response.output
    ]
    metrics: dict[str, int] = {}
    usage = response.usage
    if usage is not None:
        metrics.update(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            tokens=usage.total_tokens,
        )
        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = getattr(input_details, "cached_tokens", None)
        if cached_tokens is not None:
            metrics["prompt_cached_tokens"] = cached_tokens

    braintrust.current_span().log(
        input=[
            {"role": "system", "content": system_prompt},
            *conversation,
        ],
        output=output,
        metadata={
            "model": model,
            "tools": tools,
        },
        metrics=metrics,
    )
    return response


@traced(name="chat_turn")
def chat_turn(
    client: Any,
    conversation: list[dict[str, Any]],
    user_input: str,
    session_id: str,
    model: str = MODEL,
    system_prompt: str = SYSTEM_PROMPT,
    tools: list[dict[str, Any]] = TOOLS,
) -> str:
    """Run one agent turn, including any requested tools."""
    # braintrust.current_span().log(input=user_input, metadata={"session_id": session_id})

    for _ in range(5):
        response = call_model(
            client,
            conversation,
            model=model,
            system_prompt=system_prompt,
            tools=tools,
        )
        conversation.extend(
            item.model_dump(exclude_none=True)
            for item in response.output
        )

        tool_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        if not tool_calls:
            output = response.output_text or "The model returned no answer."
            braintrust.current_span().log(
                input=user_input,
                output=output,
                metadata={"session_id": session_id},
            )
            return output

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
            
    return "Stopped after too many tool calls."

def main() -> None:
    """Start an interactive terminal session."""
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY before running the demo.")
    if not os.getenv("BRAINTRUST_API_KEY"):
        raise SystemExit("Set BRAINTRUST_API_KEY before running the demo.")

    logger = braintrust.init_logger(project=PROJECT)

    # OpenAI tracing is implemented by call_model(), so do not let automatic
    # instrumentation create additional LLM or synthetic tool-call spans.
    # braintrust.auto_instrument(openai=False)

    from openai import OpenAI

    client = OpenAI()
    conversation: list[dict[str, Any]] = []

    print(f"Simple agent demo using {MODEL}")
    print("Try math, unit conversion, or a current-information question.")
    print("Type \\quit to exit.\n")

    user_inputs: list[str] = []
    agent_responses: list[str] = []


    session_id = str(uuid.uuid4())



    if GROUP_AS_CONVERSATION:
        with logger.start_span(
            name="Chat Session",
            metadata={"model": MODEL, "session_id": session_id},
            tags=["manual-tracing"],
        ) as chat_session:
            while True:
                try:
                    user_input = input("You: ").strip()
                    user_inputs.append(user_input)
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                if not user_input:
                    continue
                if user_input.lower() in QUIT_COMMANDS:
                    break

                conversation.append({"role": "user", "content": user_input})
                answer = chat_turn(client, conversation, user_input, session_id)
                agent_responses.append(answer)
                print(f"Agent: {answer}\n")


            chat_session.log(
                input= json.dumps(user_inputs, ensure_ascii=False),
                output= json.dumps(agent_responses, ensure_ascii=False),
                metadata={"session_id": session_id},
            )
        logger.flush()
    else:
        turn_number = 0
        while True:
            turn_number += 1
            try:
                user_input = input("You: ").strip()
                user_inputs.append(user_input)
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue
            if user_input.lower() in QUIT_COMMANDS:
                break

            conversation.append({"role": "user", "content": user_input})
            answer = chat_turn(client, conversation, user_input, session_id)
            agent_responses.append(answer)
            braintrust.current_span().log(
                metadata={"session_id": session_id, "turn": turn_number},
            )
            print(f"Agent: {answer}\n")


if __name__ == "__main__":
    main()
