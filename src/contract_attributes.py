"""Shared contract attribute normalization, independent of orchestration."""


def fixed_attributes(character: dict) -> list[str]:
    value = character.get("fixed_attributes") or []
    if isinstance(value, dict):
        other = value.get("autres") or []
        return [f"genre : {value.get('genre') or 'non précisé'}", *([str(item) for item in other] if isinstance(other, list) else [str(other)])]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return [value] if isinstance(value, str) else []
