"""Agent wrapper around the continuity checker."""

from __future__ import annotations

from src.agents.base import BaseAgent
from src.memory.continuity_checker import answer_with_evidence


class ContinuityAgent(BaseAgent):
    """Retrieve continuity evidence for a question or brief."""

    def __init__(self) -> None:
        super().__init__(name="ContinuityAgent", role="continuity")

    def run(self, input_data: dict) -> dict:
        query = (
            input_data.get("question")
            or input_data.get("brief")
            or input_data.get("scene_idea")
            or ""
        )

        result = answer_with_evidence(
            query=query,
            db_path=input_data["db_path"],
            chroma_dir=input_data["chroma_dir"],
            collection_name=input_data["collection_name"],
            n_results=input_data.get("n_results", 5),
        )
        result["agent"] = self.name
        return result
