"""Simple local LLM client abstraction with mock and Ollama modes."""

from __future__ import annotations

import json
from urllib import error, request


class LLMClient:
    """Minimal LLM client with predictable mock output."""

    def __init__(
        self,
        mode: str = "mock",
        model: str | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        timeout: float = 120.0,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.mode = mode
        self.model = model or "qwen2.5:3b"
        self.num_predict = num_predict
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.base_url = base_url.rstrip("/")

    def _generate_with_ollama(self, prompt: str) -> str:
        payload_data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        if self.num_predict is not None:
            payload_data["options"] = {"num_predict": self.num_predict}
        if self.keep_alive is not None:
            payload_data["keep_alive"] = self.keep_alive
        payload = json.dumps(payload_data).encode("utf-8")
        http_request = request.Request(
            url=f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except error.URLError as exc:
            raise RuntimeError(
                "Ollama is not available at http://localhost:11434. "
                "Make sure Ollama is running and the local API is reachable."
            ) from exc
        except TimeoutError as exc:
            raise RuntimeError("Ollama request timed out.") from exc

        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned an invalid JSON response.") from exc

        response_text = data.get("response")
        if not isinstance(response_text, str):
            raise RuntimeError("Ollama response did not contain a valid text field.")

        return response_text

    def generate(self, prompt: str) -> str:
        """Generate a response from a prompt."""
        if self.mode == "mock":
            return "[MOCK LLM RESPONSE] " + prompt[:200]
        if self.mode == "ollama":
            return self._generate_with_ollama(prompt)
        raise ValueError(f"Unsupported LLM mode: {self.mode}")
