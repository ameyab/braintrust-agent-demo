#!/usr/bin/env python3
"""Remote Braintrust eval for simple_agent.py with an LLM-simulated user."""

from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from enum import Enum
from typing import Any

from braintrust import Eval, init_dataset, traced
from braintrust.score import Score
from openai import OpenAI
from pydantic import BaseModel, Field

import simple_agent


PROJECT = os.getenv("BRAINTRUST_PROJECT", simple_agent.PROJECT)
DATASET = os.getenv("BRAINTRUST_DATASET", "Popular Usecases Test Dataset")
SIMULATED_USER_MODEL = os.getenv("SIMULATED_USER_MODEL", "gpt-4o-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")


class AgentModelParameter(BaseModel):
    value: str = Field(
        default=simple_agent.MODEL,
        description="OpenAI model used by simple_agent.py.",
    )


class SimulatedUserModelParameter(BaseModel):
    value: str = Field(
        default=SIMULATED_USER_MODEL,
        description="OpenAI model used to simulate the user.",
    )


class JudgeModelParameter(BaseModel):
    value: str = Field(
        default=JUDGE_MODEL,
        description="OpenAI model used by the optional conversation judge.",
    )


class AgentPromptParameter(BaseModel):
    value: str = Field(
        default=simple_agent.SYSTEM_PROMPT,
        description="System prompt used by simple_agent.py.",
    )


class CalculateToolDescriptionParameter(BaseModel):
    value: str = Field(
        default=simple_agent.TOOLS[0]["description"],
        description="Description for the calculate tool exposed to the agent.",
    )


class WebSearchToolDescriptionParameter(BaseModel):
    value: str = Field(
        default=simple_agent.TOOLS[1]["description"],
        description="Description for the web_search tool exposed to the agent.",
    )


class MaxTurnsParameter(BaseModel):
    value: int = Field(
        default=6,
        ge=2,
        le=20,
        description="Maximum simulated user turns per dataset example.",
    )


class TemperatureParameter(BaseModel):
    value: float = Field(
        default=0.2,
        ge=0,
        le=2,
        description="Temperature for simulated user and judge model calls.",
    )


def _param(parameters: dict[str, Any] | None, name: str, default: Any) -> Any:
    if not parameters:
        return default
    value = parameters.get(name, default)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if hasattr(value, "value"):
        return value.value
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _scenario_from_input(input: Any) -> dict[str, Any]:
    if isinstance(input, str):
        return {"goal": input, "initial_message": input}
    if not isinstance(input, dict):
        return {"goal": _text(input), "initial_message": _text(input)}

    goal = (
        input.get("goal")
        or input.get("user_goal")
        or input.get("scenario")
        or input.get("task")
        or input.get("question")
        or input.get("prompt")
        or input.get("input")
        or _text(input)
    )
    initial_message = (
        input.get("initial_message")
        or input.get("first_message")
        or input.get("question")
        or input.get("prompt")
        or input.get("input")
        or goal
    )
    return {
        "goal": _text(goal),
        "initial_message": _text(initial_message),
        "context": input.get("context") or input.get("background") or "",
        "success_criteria": input.get("success_criteria") or input.get("criteria") or "",
    }


def _criteria(input: Any, expected: Any) -> str:
    scenario = _scenario_from_input(input)
    if expected:
        return _text(expected)
    if scenario.get("success_criteria"):
        return _text(scenario["success_criteria"])
    return (
        "The assistant should satisfy the user's goal accurately, use tools when "
        "needed, avoid unsupported claims, and keep the conversation concise."
    )


class SimulatedUserTurn(BaseModel):
    message: str = Field(description="Next user message, or empty when done.")
    done: bool = Field(description="True when the conversation should stop.")


class ConversationJudgment(BaseModel):
    score: float = Field(ge=0, le=1, description="Score from 0 to 1.")
    reason: str = Field(description="Short justification for the score.")


class ToolTarget(str, Enum):
    CALCULATE = "calculate"
    WEB_SEARCH = "web_search"
    SYNTHESIS = "synthesis"


def _json_schema_format(model: type[BaseModel], name: str) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    schema["additionalProperties"] = False
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


def _json_from_model_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise json.JSONDecodeError("Could not parse model JSON", text, 0)


def _fallback_user_reply(scenario: dict[str, Any], last_assistant_message: str) -> str | None:
    """Repair common simulator failures where it mirrors the assistant."""
    lower_goal = scenario["goal"].lower()
    lower_assistant = last_assistant_message.lower()

    if (
        ("weather" in lower_goal or "forecast" in lower_goal)
        and any(word in lower_assistant for word in ["location", "city", "region", "where"])
    ):
        return "San Francisco, CA."

    if any(word in lower_assistant for word in ["clarify", "specify", "provide more", "more detail"]):
        return "Use your best judgment and make a reasonable assumption."

    return None


def _combine_clarification_with_target(
    clarification: str,
    target: ToolTarget,
    scenario: dict[str, Any],
) -> str:
    if target == ToolTarget.CALCULATE:
        if "weather" in scenario["goal"].lower() or "forecast" in scenario["goal"].lower():
            return f"{clarification} Also convert 72 degrees Fahrenheit to Celsius."
        return f"{clarification} Also calculate 18 * (7 + 3)."
    if target == ToolTarget.WEB_SEARCH:
        return f"{clarification} Also check the latest major news related to this topic."
    return clarification


def _assistant_needs_clarification(message: str) -> bool:
    lower_message = message.lower()
    return any(
        phrase in lower_message
        for phrase in (
            "please specify",
            "please provide",
            "could you provide",
            "can you provide",
            "which location",
            "what location",
            "i need",
            "need you to",
        )
    )


def _looks_like_calculate(text: str) -> bool:
    text_lower = text.lower()
    calculate_terms = (
        "calculate",
        "convert",
        "conversion",
        "how many",
        "percent",
        "percentage",
        "sum",
        "total",
        "multiply",
        "divide",
        "plus",
        "minus",
        "celsius",
        "fahrenheit",
        "kilometer",
        "kilometre",
        "mile",
    )
    return any(term in text_lower for term in calculate_terms) or any(char.isdigit() for char in text)


def _looks_like_web_search(text: str) -> bool:
    text_lower = text.lower()
    web_terms = (
        "latest",
        "current",
        "today",
        "this week",
        "news",
        "weather",
        "forecast",
        "announced",
        "announcement",
        "stock",
        "price",
        "who is",
        "where is",
    )
    return any(term in text_lower for term in web_terms)


def _target_for_second_turn(scenario: dict[str, Any]) -> ToolTarget:
    first_message = scenario["initial_message"]
    if _looks_like_calculate(first_message) and not _looks_like_web_search(first_message):
        return ToolTarget.WEB_SEARCH
    return ToolTarget.CALCULATE


def _should_add_synthesis_turn(scenario: dict[str, Any]) -> bool:
    key = scenario["goal"] + scenario["initial_message"]
    return sum(ord(char) for char in key) % 3 == 0


def _fallback_targeted_reply(
    target: ToolTarget,
    scenario: dict[str, Any],
    transcript: list[dict[str, str]],
) -> str:
    if target == ToolTarget.CALCULATE:
        if "weather" in scenario["goal"].lower() or "forecast" in scenario["goal"].lower():
            return "Convert 72 degrees Fahrenheit to Celsius so I can compare it with the forecast."
        return "Also calculate 18 * (7 + 3)."

    if target == ToolTarget.WEB_SEARCH:
        return "Now check the latest major news related to this topic."

    user_questions = [message["content"] for message in transcript if message["role"] == "user"]
    if len(user_questions) >= 2:
        return (
            "Using your answers to my first two questions, give me one practical "
            "recommendation that connects them."
        )
    return "Use the previous answers together and give me one practical recommendation."


def _is_reflected_clarification(message: str, last_assistant_message: str) -> bool:
    message_lower = message.lower().strip()
    assistant_lower = last_assistant_message.lower()
    reflection_starts = (
        "please provide",
        "could you please provide",
        "can you provide",
        "please specify",
        "could you specify",
        "i need you to",
    )
    return (
        message_lower.startswith(reflection_starts)
        and any(word in assistant_lower for word in ["provide", "specify", "clarify", "need"])
    )


def _agent_tools(
    calculate_description: str,
    web_search_description: str,
) -> list[dict[str, Any]]:
    tools = deepcopy(simple_agent.TOOLS)
    for tool in tools:
        if tool["name"] == "calculate":
            tool["description"] = calculate_description
        elif tool["name"] == "web_search":
            tool["description"] = web_search_description
    return tools


@traced(name="simulated_user", type="llm")
def simulated_user_turn(
    client: OpenAI,
    *,
    model: str,
    temperature: float,
    scenario: dict[str, Any],
    transcript: list[dict[str, str]],
    required_target: ToolTarget | None = None,
) -> dict[str, Any]:
    required_target_instruction = ""
    if required_target == ToolTarget.CALCULATE:
        required_target_instruction = (
            "\nMandatory next-turn objective: ask a natural follow-up that requires "
            "the assistant to use the calculate tool for arithmetic or unit conversion. "
            "Do not end the conversation on this turn."
        )
    elif required_target == ToolTarget.WEB_SEARCH:
        required_target_instruction = (
            "\nMandatory next-turn objective: ask a natural follow-up that requires "
            "the assistant to use the web_search tool for current information. "
            "Do not end the conversation on this turn."
        )
    elif required_target == ToolTarget.SYNTHESIS:
        required_target_instruction = (
            "\nMandatory next-turn objective: ask a natural third question that uses "
            "the assistant's previous answers together in a meaningful way. "
            "This should be a synthesis question, not a repeat of either earlier question."
        )

    response = client.responses.create(
        model=model,
        temperature=temperature,
        instructions=(
            "You are simulating the USER in a terminal chat with an AI assistant. "
            "You are not the assistant and you must not tell the assistant what to provide. "
            "Your job is to pursue the scenario goal like a cooperative human user.\n\n"
            "Rules:\n"
            "- If the assistant asks a clarifying question, answer it directly with the missing detail.\n"
            "- If the scenario does not contain that detail, invent a plausible concrete detail and keep it consistent.\n"
            "- Do not repeat or paraphrase the assistant's clarification request.\n"
            "- Do not ask the assistant to provide information that the assistant just asked you for.\n"
            "- Ask at most one short follow-up only when the previous assistant answer is incomplete or wrong.\n"
            "- Set done=true when the assistant has satisfied the goal, refuses due to missing capability, or the conversation is stuck.\n"
            "- For weather/location ambiguity, a valid clarification answer is a concrete place like San Francisco, CA.\n"
            "- When done=true, set message to an empty string unless a natural short acknowledgement is useful."
            f"{required_target_instruction}"
        ),
        input=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "scenario": scenario,
                        "transcript": transcript,
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        text=_json_schema_format(SimulatedUserTurn, "simulated_user_turn"),
    )
    output = SimulatedUserTurn.model_validate(
        _json_from_model_text(response.output_text or "{}")
    )
    message = output.message.strip()
    done = output.done
    if transcript:
        fallback = _fallback_user_reply(scenario, transcript[-1]["content"])
        if fallback and _is_reflected_clarification(message, transcript[-1]["content"]):
            message = fallback
            done = False
    if required_target and (done or not message):
        message = _fallback_targeted_reply(required_target, scenario, transcript)
        done = False
    return {
        "message": message,
        "done": done,
    }


def run_conversation(input: Any, hooks: Any) -> dict[str, Any]:
    parameters = hooks.parameters or {}
    agent_model = _param(parameters, "agent_model", simple_agent.MODEL)
    simulated_user_model = _param(parameters, "simulated_user_model", SIMULATED_USER_MODEL)
    judge_model = _param(parameters, "judge_model", JUDGE_MODEL)
    agent_prompt = _param(parameters, "agent_prompt", simple_agent.SYSTEM_PROMPT)
    calculate_tool_description = _param(
        parameters,
        "calculate_tool_description",
        simple_agent.TOOLS[0]["description"],
    )
    web_search_tool_description = _param(
        parameters,
        "web_search_tool_description",
        simple_agent.TOOLS[1]["description"],
    )
    max_turns = max(2, int(_param(parameters, "max_turns", 6)))
    temperature = float(_param(parameters, "temperature", 0.2))
    tools = _agent_tools(calculate_tool_description, web_search_tool_description)

    client = OpenAI()
    conversation: list[dict[str, Any]] = []
    transcript: list[dict[str, str]] = []
    scenario = _scenario_from_input(input)
    session_id = str(uuid.uuid4())
    second_turn_target = _target_for_second_turn(scenario)
    desired_user_turns = 3 if max_turns >= 3 and _should_add_synthesis_turn(scenario) else 2
    exercised_second_tool = False

    user_message = scenario["initial_message"]

    for turn in range(1, max_turns + 1):
        if not user_message:
            break

        conversation.append({"role": "user", "content": user_message})
        transcript.append({"role": "user", "content": user_message})
        answer = simple_agent.chat_turn(
            client,
            conversation,
            user_message,
            session_id,
            model=agent_model,
            system_prompt=agent_prompt,
            tools=tools,
        )
        transcript.append({"role": "assistant", "content": answer})

        hooks.report_progress(
            {
                "progress": turn / max_turns,
                "data": {"turn": turn, "last_user_message": user_message},
            }
        )

        if turn >= max_turns:
            break

        required_target = None
        needs_clarification = _assistant_needs_clarification(answer)
        if needs_clarification:
            desired_user_turns = min(max_turns, max(desired_user_turns, turn + 2))
            if turn + 2 > max_turns and not exercised_second_tool:
                required_target = second_turn_target
                exercised_second_tool = True
        elif not exercised_second_tool:
            required_target = second_turn_target
            exercised_second_tool = True
        elif turn == 2 and desired_user_turns >= 3:
            required_target = ToolTarget.SYNTHESIS
        elif turn >= desired_user_turns:
            break

        next_user = simulated_user_turn(
            client,
            model=simulated_user_model,
            temperature=temperature,
            scenario=scenario,
            transcript=transcript,
            required_target=required_target,
        )
        if next_user["done"] and turn >= desired_user_turns:
            break
        if needs_clarification and required_target:
            fallback = _fallback_user_reply(scenario, answer)
            if fallback:
                user_message = _combine_clarification_with_target(
                    fallback,
                    required_target,
                    scenario,
                )
                continue
        user_message = next_user["message"]

    if hooks.metadata is not None:
        hooks.metadata.update(
            {
                "session_id": session_id,
                "agent_model": agent_model,
                "simulated_user_model": simulated_user_model,
                "judge_model": judge_model,
                "agent_prompt": agent_prompt,
                "calculate_tool_description": calculate_tool_description,
                "web_search_tool_description": web_search_tool_description,
                "turns": len([m for m in transcript if m["role"] == "user"]),
            }
        )

    return {
        "transcript": transcript,
        "final_answer": next(
            (message["content"] for message in reversed(transcript) if message["role"] == "assistant"),
            "",
        ),
        "_judge_model": judge_model,
        "_agent_prompt": agent_prompt,
        "_tools": tools,
    }


@traced(name="conversation_judge", type="llm")
def conversation_quality(input: Any, output: dict[str, Any], expected: Any) -> Score:
    if not os.getenv("OPENAI_API_KEY"):
        return Score(
            name="conversation_quality",
            score=None,
            metadata={"reason": "OPENAI_API_KEY is not set; judge skipped."},
        )

    client = OpenAI()
    criteria = _criteria(input, expected)
    response = client.responses.create(
        model=output.get("_judge_model", JUDGE_MODEL),
        temperature=0,
        instructions=(
            "You are an evaluator for a tool-using terminal assistant. "
            "Grade whether the transcript satisfies the criteria. "
            "Score is 0 to 1."
        ),
        input=[
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "criteria": criteria,
                        "transcript": output.get("transcript", []),
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        text=_json_schema_format(ConversationJudgment, "conversation_judgment"),
    )
    judged = ConversationJudgment.model_validate(
        _json_from_model_text(response.output_text or "{}")
    )
    return Score(
        name="conversation_quality",
        score=judged.score,
        metadata={"reason": judged.reason, "criteria": criteria},
    )


Eval(
    PROJECT,
    data=init_dataset(project=PROJECT, name=DATASET),
    task=run_conversation,
    scores=[conversation_quality],
    experiment_name=os.getenv("BRAINTRUST_EXPERIMENT", "simple-agent-remote-eval"),
    max_concurrency=int(os.getenv("BRAINTRUST_EVAL_MAX_CONCURRENCY", "2")),
    metadata={"dataset": DATASET, "eval_type": "simulated_user_conversation"},
    tags=["remote-eval", "simulated-user"],
    parameters={
        "agent_model": AgentModelParameter,
        "simulated_user_model": SimulatedUserModelParameter,
        "judge_model": JudgeModelParameter,
        "agent_prompt": AgentPromptParameter,
        "calculate_tool_description": CalculateToolDescriptionParameter,
        "web_search_tool_description": WebSearchToolDescriptionParameter,
        "max_turns": MaxTurnsParameter,
        "temperature": TemperatureParameter,
    },
)
