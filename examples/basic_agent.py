"""Minimal example: a deep agent on GigaChat.

The `deepagents-gigachat` package registers its harness profile via the
`deepagents.harness_profiles` entry point, so `deepagents` picks it up
automatically once the package is installed.

Run from the repository root:

    uv run python examples/basic_agent.py
"""

from __future__ import annotations

import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_gigachat import GigaChat


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def main() -> None:
    load_dotenv()

    model = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra"),
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )
    agent = create_deep_agent(model=model, tools=[add])

    question = "Use the `add` tool to compute 21 + 21 and report the result."
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
