"""Shared helpers for agent depth configuration."""

from __future__ import annotations


def get_agent_strategy_summary(agent_depth: str) -> str:
    if agent_depth == "fast":
        return "agents mostly deterministic, LLM mainly for stylist"
    if agent_depth == "deep":
        return "reserved for deeper LLM-based agent deliberation"
    return "default writer room with deterministic analysis agents and LLM stylist"
