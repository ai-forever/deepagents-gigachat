"""Minimal example: a deep agent on GigaChat.

What happens:
1. Register the GigaChat harness profile through
   `deepagents_gigachat.register_harness()`. After that, `deepagents`
   uses our system prompt, overridden tool descriptions (`read_file`,
   `write_file`, `grep`, `execute`, and so on), and adds the `think` tool.
2. Create a `GigaChat` model through `langchain-gigachat`.
3. Add two small Python functions as agent tools.
4. Wrap the model with `create_deep_agent` from `deepagents`. The profile
   is picked up automatically from the model provider (`giga`).

Registration note: the `deepagents-gigachat` package declares the
`deepagents.harness_profiles` entry point, so `deepagents` can call
`register_harness()` automatically on first use. This example calls it
explicitly so the package dependency is visible in the code. Repeated calls
are safe.

Run from the repository root:

    uv run python examples/basic_agent.py
"""

from __future__ import annotations

import os
from pathlib import Path

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain_gigachat import GigaChat

from deepagents_gigachat import register_harness

register_harness()


def explain_python_topic(topic: str) -> str:
    """Explain one Python topic in simple words for a beginner."""
    explanations = {
        "function": (
            "A function is a named piece of a program. "
            "You describe the steps once, then call them by name."
        ),
        "list": (
            "A list is a box where values are stored in order. "
            "For example: ['apple', 'banana', 'pear']."
        ),
        "loop": (
            "A loop repeats the same action several times. "
            "For example, you can go through every word in a list."
        ),
    }
    return explanations.get(
        topic.lower(),
        "This is a Python topic. Try explaining it with a tiny code example.",
    )


def make_practice_task(topic: str, minutes: int) -> str:
    """Create a short Python practice task for the given number of minutes."""
    return (
        f"{minutes}-minute task: create an example about '{topic}', "
        "write 5-10 lines of code, and run the program. Then change one value "
        "and see how the result changes."
    )


def build_agent() -> object:
    """Build GigaChat and wrap it in a deep agent."""
    model = GigaChat(
        model=os.getenv("GIGACHAT_MODEL", "GigaChat-3-Ultra"),
        base_url=os.getenv("GIGACHAT_BASE_URL", "https://gigachat.sberdevices.ru/v1"),
        verify_ssl_certs=False,
        profanity_check=False,
        timeout=600,
    )
    return create_deep_agent(model=model, tools=[explain_python_topic, make_practice_task])


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

    if not os.getenv("GIGACHAT_CREDENTIALS") and not (
        os.getenv("GIGACHAT_USER") and os.getenv("GIGACHAT_PASSWORD")
    ):
        raise SystemExit(
            "GigaChat credentials are not configured. "
            "Set GIGACHAT_CREDENTIALS or the GIGACHAT_USER + GIGACHAT_PASSWORD pair."
        )

    agent = build_agent()

    question = (
        "I am learning Python. Use your tools: explain what a function is, "
        "and create a short 15-minute practice task."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    last_message = result["messages"][-1]
    print("=" * 60)
    print("Question:", question)
    print("-" * 60)
    print("Answer:", last_message.content)
    print("=" * 60)


if __name__ == "__main__":
    main()
