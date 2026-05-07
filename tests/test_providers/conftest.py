import json
from urllib import error


class FakeHTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def close(self) -> None:
        return None


def patch_success(monkeypatch, payload: object) -> None:
    monkeypatch.setattr(
        "src.llm.providers._http.request.urlopen",
        lambda request, timeout: FakeHTTPResponse(payload),
    )


def patch_http_error(monkeypatch, status: int) -> None:
    def raise_http_error(request, timeout):
        raise error.HTTPError(
            url=request.full_url,
            code=status,
            msg="error",
            hdrs={},
            fp=FakeHTTPResponse({"error": "provider error"}),
        )

    monkeypatch.setattr("src.llm.providers._http.request.urlopen", raise_http_error)


def patch_timeout(monkeypatch) -> None:
    def raise_timeout(request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("src.llm.providers._http.request.urlopen", raise_timeout)
