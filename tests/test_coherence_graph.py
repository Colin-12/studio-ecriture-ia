from pathlib import Path

from src.graph.coherence_graph import CoherenceGraph


def test_coherence_graph_save_and_load_roundtrip(tmp_path: Path) -> None:
    graph = CoherenceGraph()
    graph.add_character("Alice", role="protagonist")
    graph.add_location("Paris", country="France")
    graph.add_event("event-1", "Alice arrives in Paris.", chapter_number=1)
    graph.add_relationship("Alice", "Paris", "travels_to", chapter_number=1)
    graph.add_relationship("event-1", "Paris", "happens_in", chapter_number=1)

    graph_path = tmp_path / "coherence_graph.json"
    graph.save_json(graph_path)

    loaded_graph = CoherenceGraph.load_json(graph_path)

    assert loaded_graph.graph.nodes["Alice"]["node_type"] == "character"
    assert loaded_graph.graph.nodes["Alice"]["role"] == "protagonist"
    assert loaded_graph.graph.nodes["Paris"]["node_type"] == "location"
    assert loaded_graph.graph.nodes["event-1"]["description"] == "Alice arrives in Paris."

    edges = list(loaded_graph.graph.edges(data=True))
    assert len(edges) == 2
    assert any(edge_data["label"] == "travels_to" for _, _, edge_data in edges)
    assert any(edge_data["label"] == "happens_in" for _, _, edge_data in edges)
