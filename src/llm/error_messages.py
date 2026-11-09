"""French generation errors and credential-safe technical details."""

from __future__ import annotations

import os
import re
import traceback
from urllib.parse import parse_qsl, urlsplit

from src.llm.exceptions import PrepayCreditsDepletedError, SpendCapExceededError

PREPAY_MESSAGE = "Crédits Gemini épuisés : recharge sur AI Studio, page Facturation (https://ai.studio/projects)."
SPEND_MESSAGE = "Plafond mensuel de dépense Gemini atteint : relève le plafond ou attends le 1er du mois."
NETWORK_MESSAGE = "Un fournisseur est inaccessible depuis ce réseau : coupe le VPN ou change de réseau."
KNOWN_ERROR_MESSAGES = {PREPAY_MESSAGE, SPEND_MESSAGE, NETWORK_MESSAGE}


def exception_chain(exc: BaseException) -> list[BaseException]:
    chain = []
    visited: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        chain.append(current)
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    return chain


def is_network_denial(exc: BaseException) -> bool:
    text = " ".join(str(error) for error in exception_chain(exc)).casefold()
    return "access denied" in text and not re.search(r"\b(?:key|api[_ -]?key|clé|cle|credential|credentials|token)\b", text)


def redact_error_text(text: str) -> str:
    text = text.replace("\\/", "/")
    secrets = [value for name, value in os.environ.items() if value and re.search(r"(?:API_?KEY|TOKEN|SECRET|PASSWORD)$", name, re.I)]

    def redact_url(match: re.Match) -> str:
        url = match.group()
        try:
            parsed = urlsplit(url)
            sensitive = any(re.search(r"key|token|secret|password|credential|signature", name, re.I) for name, _ in parse_qsl(parsed.query))
            if parsed.username or parsed.password or sensitive or any(secret in url for secret in secrets) or re.search(r"AIza[\w-]+|gsk_[\w-]+|sk-[\w-]+", url):
                return "[URL contenant des identifiants masquée]"
        except ValueError:
            return "[URL masquée]"
        return url

    text = re.sub(r"https?://[^\s<>\"']+", redact_url, text)
    for value in secrets:
        text = text.replace(value, "[MASQUÉ]")
    text = re.sub(r'''(?i)((?:[\w-]*api[_ -]?key|[\w-]*token|secret|password|authorization)["']?\s*[:=]\s*["']?)(?:Bearer\s+)?[^\s,"'}]+''', r"\1[MASQUÉ]", text)
    text = re.sub(r"\b(?:AIza[\w-]+|gsk_[\w-]+|sk-[\w-]+)\b", "[MASQUÉ]", text)
    return text


def format_generation_error(exc: BaseException) -> str:
    chain = exception_chain(exc)
    if any(isinstance(error, PrepayCreditsDepletedError) for error in chain):
        return PREPAY_MESSAGE
    if any(isinstance(error, SpendCapExceededError) for error in chain):
        return SPEND_MESSAGE
    if is_network_denial(exc):
        return NETWORK_MESSAGE
    return redact_error_text(str(exc))


def generation_error_details(exc: BaseException) -> str:
    return redact_error_text("".join(traceback.format_exception(exc)))
