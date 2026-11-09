"""Shared contract attribute normalization, independent of orchestration."""

from copy import deepcopy


def fixed_attributes(character: dict) -> list[str]:
    value = character.get("fixed_attributes") or []
    if isinstance(value, dict):
        other = value.get("autres") or []
        return [f"genre : {value.get('genre') or 'non précisé'}", *([str(item) for item in other] if isinstance(other, list) else [str(other)])]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return [value] if isinstance(value, str) else []


def attribute_fields(character: dict) -> dict[str, str]:
    values = {"genre": "non précisé", "âge": "", "profession": ""}
    for attr in fixed_attributes(character):
        key, colon, value = attr.partition(":")
        key = key.strip().casefold()
        key = "âge" if key == "age" else key
        if colon and key in values:
            values[key] = value.strip().casefold() if key == "genre" else value.strip()
    return values


def with_attribute_fields(character: dict, genre: str, age: str, profession: str) -> dict:
    if genre not in {"non précisé", "masculin", "féminin", "autre"}:
        raise ValueError("Genre invalide")
    result = deepcopy(character)
    attrs = [attr for attr in fixed_attributes(character) if attr.partition(":")[0].strip().casefold() not in {"genre", "âge", "age", "profession"}]
    attrs.append(f"genre : {genre}")
    if age.strip():
        attrs.append(f"âge : {age.strip()}")
    if profession.strip():
        attrs.append(f"profession : {profession.strip()}")
    result["fixed_attributes"] = attrs
    return result


def contract_with_characters(contract: dict, characters: list[dict]) -> dict:
    result = deepcopy(contract)
    known = result.get("characters") or []
    result["characters"] = known
    for character in characters:
        existing = next((ch for ch in known if ch.get("name", "").casefold() == character["name"].casefold()), None)
        if existing is None:
            known.append(deepcopy(character))
        else:
            existing["fixed_attributes"] = fixed_attributes(character)
    return result


def without_attribute_statements(threads: list[str], characters: list[dict]) -> list[str]:
    statements = set()
    for ch in characters:
        name = ch["name"].casefold()
        values = attribute_fields(ch)
        if values["genre"] in {"masculin", "féminin"}:
            statements.add(f"{name} est {'un homme' if values['genre'] == 'masculin' else 'une femme'}")
        if values["âge"]:
            statements.add(f"{name} a {values['âge'].casefold()}")
        if values["profession"]:
            statements.add(f"{name} est {values['profession'].casefold()}")
    return [thread for thread in threads if " ".join(thread.casefold().strip(" .!").split()) not in statements]
