"""Base class for minimal deterministic agents."""

from __future__ import annotations


class BaseAgent:
    """Base class for simple agent wrappers."""

    def __init__(self, name: str, role: str) -> None:
        self.name = name
        self.role = role

    def run(self, input_data: dict) -> dict:
        """Run the agent on a dictionary input."""
        raise NotImplementedError
