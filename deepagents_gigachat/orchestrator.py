"""Generic routed execution helpers shared by benchmark adapters."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from deepagents_gigachat.routing import RoutingDecision


def build_deep_agent_env(
    *,
    model_name: str,
    deep_profile: str | None,
) -> dict[str, str | None]:
    """Build environment overrides for the deepagents execution branch."""
    env: dict[str, str | None] = {"GIGACHAT_MODEL": model_name}
    if deep_profile is not None:
        env["DEEPAGENTS_GIGACHAT_PROFILE"] = deep_profile
    return env


@contextmanager
def temporary_env(overrides: Mapping[str, str | None]) -> Iterator[None]:
    """Temporarily apply environment overrides and then restore previous values."""
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def run_routed[TTask, TReturn](
    task: TTask,
    *,
    decision: RoutingDecision,
    direct_runner: Callable[..., TReturn],
    deep_runner: Callable[..., TReturn],
    direct_kwargs: Mapping[str, Any] | None = None,
    deep_kwargs: Mapping[str, Any] | None = None,
    deep_env: Mapping[str, str | None] | None = None,
) -> TReturn:
    """Dispatch a task between direct and deep execution branches."""
    if decision.execution_route == "direct":
        return direct_runner(task, **dict(direct_kwargs or {}))

    if not deep_env:
        return deep_runner(task, **dict(deep_kwargs or {}))

    with temporary_env(deep_env):
        return deep_runner(task, **dict(deep_kwargs or {}))


__all__ = [
    "build_deep_agent_env",
    "run_routed",
    "temporary_env",
]
