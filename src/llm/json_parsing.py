"""JSON response parsing shared independently of agents and memory."""

import json


def parse_json_response(text: str) -> object:
    """Accept fenced JSON and a JSON object/array embedded in commentary."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
            return value
        except ValueError:
            continue
    raise ValueError("aucun JSON lisible")
