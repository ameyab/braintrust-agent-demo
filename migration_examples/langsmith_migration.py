#!/usr/bin/env python3
"""Tiny support chatbot that traces each run to Braintrust and LangSmith."""

from __future__ import annotations

import asyncio
import inspect
import os
import uuid
from typing import Any

# LangSmith tracing is opt-in via env; default it on for this migration demo.
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")

import braintrust
import langsmith as ls
from braintrust import traced as braintrust_traced
from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-5-nano")
BRAINTRUST_PROJECT = os.getenv("BRAINTRUST_PROJECT", "Support Agent Migration")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", BRAINTRUST_PROJECT)
QUIT_COMMANDS = {"\\q", "\\quit", "\\exit", "quit", "exit"}

SYSTEM_PROMPT = """
You are a helpful customer support agent for an e-commerce company.
Be empathetic but efficient. Ask one clarifying question when needed.
Resolve simple order, return, shipping, refund, and account questions directly.
If escalation is needed, explain why and what the customer should expect next.
""".strip()


def _strip_client(inputs: dict[str, Any]) -> dict[str, Any]:
    """Keep trace inputs serializable by removing the OpenAI client object."""
    return {key: value for key, value in inputs.items() if key != "client"}


def _chat_turn_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    conversation = inputs.get("conversation") or []
    return {
        "user_input": inputs.get("user_input"),
        "session_id": inputs.get("session_id"),
        "model": inputs.get("model"),
        "conversation_length": len(conversation),
    }


def _usage_metrics(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}

    metrics: dict[str, int] = {}
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens
    if completion_tokens is not None:
        metrics["completion_tokens"] = completion_tokens
    if total_tokens is not None:
        metrics["tokens"] = total_tokens
    return metrics


def _dump_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)
    return {"response": str(response)}


def _chat_completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if not model.startswith("gpt-5"):
        kwargs["temperature"] = 0.2
    return kwargs


@braintrust_traced(
    name="openai.chat.completions.create",
    type="llm",
    notrace_io=True,
)
@ls.traceable(
    name="openai.chat.completions.create",
    run_type="llm",
    project_name=LANGSMITH_PROJECT,
    process_inputs=_strip_client,
)
def call_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    model: str = MODEL,
) -> Any:
    """Call OpenAI once and record comparable LLM spans in both systems."""
    request = _chat_completion_kwargs(model, messages)
    response = client.chat.completions.create(**request)

    braintrust.current_span().log(
        input=request,
        output=_dump_response(response),
        metadata={"model": model},
        metrics=_usage_metrics(response),
    )
    return response


@braintrust_traced(name="support_agent_chat_turn")
@ls.traceable(
    name="support_agent_chat_turn",
    run_type="chain",
    project_name=LANGSMITH_PROJECT,
    process_inputs=_chat_turn_inputs,
)
def chat_turn(
    client: OpenAI,
    conversation: list[dict[str, str]],
    user_input: str,
    session_id: str,
    model: str = MODEL,
) -> str:
    """Run one customer-support turn and append it to the conversation."""
    conversation.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *conversation]
    response = call_model(client, messages, model=model)

    answer = response.choices[0].message.content or ""
    conversation.append({"role": "assistant", "content": answer})

    braintrust.current_span().log(
        input=user_input,
        output=answer,
        metadata={"model": model, "session_id": session_id},
    )
    return answer


def flush_langsmith() -> None:
    """Flush LangSmith's background queue before this terminal process exits."""
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

    braintrust_logger = braintrust.init_logger(project=BRAINTRUST_PROJECT)
    client = OpenAI()
    conversation: list[dict[str, str]] = []
    session_id = str(uuid.uuid4())
    user_inputs: list[str] = []
    agent_outputs: list[str] = []

    print(f"Support agent migration demo using {MODEL}")
    print(f"Braintrust project: {BRAINTRUST_PROJECT}")
    print(f"LangSmith project: {LANGSMITH_PROJECT}")
    print("Type \\quit to exit.\n")

    try:
        with braintrust_logger.start_span(
            name="support_agent_chat_session",
            metadata={
                "model": MODEL,
                "session_id": session_id,
                "langsmith_project": LANGSMITH_PROJECT,
            },
            tags=["langsmith-migration", "support-agent"],
        ) as braintrust_session:
            with ls.trace(
                "support_agent_chat_session",
                "chain",
                project_name=LANGSMITH_PROJECT,
                inputs={"session_id": session_id},
            ) as langsmith_session:
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

                session_output = (
                    agent_outputs[-1] if agent_outputs else "No support turns completed."
                )
                braintrust_session.log(
                    input=user_inputs,
                    output=session_output,
                    metadata={
                        "session_id": session_id,
                        "turn_count": len(agent_outputs),
                    },
                )
                langsmith_session.end(
                    outputs={
                        "last_response": session_output,
                        "turn_count": len(agent_outputs),
                    }
                )
    finally:
        braintrust_logger.flush()
        flush_langsmith()


if __name__ == "__main__":
    main()
