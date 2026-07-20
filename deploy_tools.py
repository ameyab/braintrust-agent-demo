#!/usr/bin/env python3
"""Deployable Braintrust definitions for the demo agent's tools.

Push with:
    bt functions push deploy_tools.py --runner .venv/bin/python \
        --requirements requirements.txt --if-exists replace --yes
"""

from __future__ import annotations

import braintrust
from pydantic import BaseModel, ConfigDict, Field

from agent import PROJECT, TOOLS, calculate, web_search

TOOL_DEFINITIONS = {tool["name"]: tool for tool in TOOLS}


class CalculateParameters(BaseModel):
    """Input schema for the calculator tool."""

    model_config = ConfigDict(extra="forbid")

    expression: str = Field(
        description=TOOL_DEFINITIONS["calculate"]["parameters"]["properties"][
            "expression"
        ]["description"]
    )


class WebSearchParameters(BaseModel):
    """Input schema for the web-search tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description=TOOL_DEFINITIONS["web_search"]["parameters"]["properties"]["query"][
            "description"
        ]
    )


project = braintrust.projects.create(PROJECT)
project.tools.create(
    handler=calculate,
    name="calculate",
    description=TOOL_DEFINITIONS["calculate"]["description"],
    parameters=CalculateParameters,
    if_exists="replace",
)
project.tools.create(
    handler=web_search,
    name="web_search",
    description=TOOL_DEFINITIONS["web_search"]["description"],
    parameters=WebSearchParameters,
    if_exists="replace",
)
