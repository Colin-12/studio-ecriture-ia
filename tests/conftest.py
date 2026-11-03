"""Keep every test's LLM usage records outside the real application log."""

import hashlib
from pathlib import Path

import pytest

REAL_USAGE_LOG = Path(__file__).resolve().parents[1] / "logs/llm_usage.jsonl"
REAL_GEMINI_ERROR_LOG = Path(__file__).resolve().parents[1] / "logs/gemini_errors.jsonl"


def _real_log_fingerprint(path=REAL_USAGE_LOG):
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def real_usage_log_unchanged():
    before = _real_log_fingerprint()
    gemini_before = _real_log_fingerprint(REAL_GEMINI_ERROR_LOG)
    yield
    assert _real_log_fingerprint() == before, "Tests modified the real LLM usage log"
    assert _real_log_fingerprint(REAL_GEMINI_ERROR_LOG) == gemini_before, "Tests modified the real Gemini error log"


@pytest.fixture(autouse=True)
def isolate_llm_usage_log(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_USAGE_LOG", str(tmp_path / "logs/llm_usage.jsonl"))
    monkeypatch.setenv("GEMINI_ERROR_LOG", str(tmp_path / "logs/gemini_errors.jsonl"))
