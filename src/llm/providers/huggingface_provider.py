"""Hugging Face Inference API provider."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

from src.llm.base import LLMProvider, LLMResponse
from src.llm.exceptions import InvalidProviderResponseError
from src.llm.providers._http import post_json


class HuggingFaceProvider(LLMProvider):
    provider_name = "huggingface"

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.3") -> None:
        load_dotenv()
        self.model = model
        self.api_key = os.getenv("HF_API_KEY")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        response_format: str | None = None,
        timeout: int = 120,
    ) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False,
            },
        }
        data, latency_ms = post_json(
            f"https://api-inference.huggingface.co/models/{quote(self.model, safe='')}",
            payload,
            headers={"Authorization": f"Bearer {self.api_key or ''}"},
            timeout=timeout,
        )
        text = _parse_response(data)
        raw = {"response": data} if not isinstance(data, dict) else data
        return LLMResponse(text, self.provider_name, self.model, None, None, latency_ms, raw)

    def is_available(self) -> bool:
        return bool(self.api_key)


def _parse_response(data: Any) -> str:
    if isinstance(data, list) and data and isinstance(data[0], dict):
        generated = data[0].get("generated_text")
        if isinstance(generated, str):
            return generated
    if not isinstance(data, dict):
        raise InvalidProviderResponseError("Hugging Face response is missing text.")
    generated = data.get("generated_text")
    if isinstance(generated, str):
        return generated
    if isinstance(data.get("error"), str):
        raise InvalidProviderResponseError(data["error"])
    raise InvalidProviderResponseError("Hugging Face response is missing text.")
