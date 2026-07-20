#!/usr/bin/env python3
"""Deploy the demo agent's system prompt to Braintrust.

Push with:
    bt functions push deploy_prompt.py --runner .venv/bin/python \
        --if-exists replace --yes --environment production
"""

from __future__ import annotations

import braintrust

from agent import MODEL, PROJECT, SYSTEM_PROMPT

project = braintrust.projects.create(PROJECT)
project.prompts.create(
    name="Terminal Assistant",
    slug="terminal-assistant",
    description="System prompt used by the demo terminal agent.",
    messages=[{"role": "system", "content": SYSTEM_PROMPT}],
    model=MODEL,
    if_exists="replace",
)
