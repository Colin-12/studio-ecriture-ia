"""Gemini model catalog and cached output limits from the models list endpoint."""

import json
from functools import lru_cache
from typing import Any, Callable
from urllib import error, parse, request

from src.llm.exceptions import ProviderUnavailableError


def _get(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = request.Request(url, headers={"User-Agent": "studio-ecriture-ia/1.0", **headers})
    with request.urlopen(req, timeout=20) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise ValueError("Invalid Gemini catalog")
    return data


def fetch_gemini_models(
    api_key: str,
    get_json: Callable[[str, dict[str, str]], dict[str, Any]] = _get,
) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    page = ""
    while True:
        url = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000"
        if page:
            url += "&pageToken=" + parse.quote(page, safe="")
        data = get_json(url, {"x-goog-api-key": api_key})
        for model in data.get("models", []):
            if "generateContent" in model.get("supportedGenerationMethods", []):
                models[model["name"].removeprefix("models/")] = model
        page = data.get("nextPageToken", "")
        if not page:
            return models


@lru_cache(maxsize=8)
def _catalog(api_key: str) -> dict[str, dict[str, Any]]:
    return fetch_gemini_models(api_key)


def output_token_limit(model: str, api_key: str) -> int:
    try:
        limit = _catalog(api_key).get(model, {}).get("outputTokenLimit")
    except (error.URLError, OSError, ValueError, KeyError) as exc:
        raise ProviderUnavailableError("Gemini model catalog unavailable; cannot determine outputTokenLimit.") from exc
    if type(limit) is not int or limit <= 0:
        raise ProviderUnavailableError(f"Gemini outputTokenLimit unavailable for {model}.")
    return limit
