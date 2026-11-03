"""Keep every test's LLM usage records outside the real application log."""

import hashlib
from pathlib import Path

import pytest

REAL_USAGE_LOG = Path(__file__).resolve().parents[1] / "logs/llm_usage.jsonl"


def _real_log_fingerprint():
    if not REAL_USAGE_LOG.exists():
        return None
    return hashlib.sha256(REAL_USAGE_LOG.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def real_usage_log_unchanged():
    before = _real_log_fingerprint()
    yield
    assert _real_log_fingerprint() == before, "Tests modified the real LLM usage log"


@pytest.fixture(autouse=True)
def isolate_llm_usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USAGE_LOG", str(tmp_path / "logs/llm_usage.jsonl"))
