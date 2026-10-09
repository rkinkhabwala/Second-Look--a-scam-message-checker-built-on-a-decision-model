import pytest
import respx


@pytest.fixture(autouse=True)
def respx_router():
    """Fail any HTTP request a test didn't explicitly mock. Tests never call the real API."""
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        yield router


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
