"""Simple coherence graph backed by NetworkX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
from networkx.readwrite import json_graph


class CoherenceGraph:
    """Thin wrapper around a serializable multi-directed graph."""

    def __init__(self) -> None:
        self.graph = nx.MultiDiGraph()

    def add_character(self, name: str, **attrs: Any) -> None:
        self.graph.add_node(name, node_type="character", **attrs)

    def add_location(self, name: str, **attrs: Any) -> None:
        self.graph.add_node(name, node_type="location", **attrs)

    def add_event(
        self,
        event_id: str,
        description: str,
        chapter_number: int,
        **attrs: Any,
    ) -> None:
        self.graph.add_node(
            event_id,
            node_type="event",
            description=description,
            chapter_number=chapter_number,
            **attrs,
        )

    def add_relationship(
        self,
        source: str,
        target: str,
        label: str,
        chapter_number: int,
        **attrs: Any,
    ) -> None:
        self.graph.add_edge(
            source,
            target,
            label=label,
            chapter_number=chapter_number,
            **attrs,
        )

    def save_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json_graph.node_link_data(self.graph, edges="edges")
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "CoherenceGraph":
        input_path = Path(path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        instance = cls()
        instance.graph = json_graph.node_link_graph(
            payload,
            edges="edges",
            multigraph=True,
            directed=True,
        )
        return instance
