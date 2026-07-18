"""Small offline tests for the demo's response shaping."""

from unittest.mock import Mock, patch

from agent import calculate, web_search


def test_calculate() -> None:
    response = Mock()
    response.text = "30"
    response.raise_for_status.return_value = None

    with patch("agent.requests.get", return_value=response) as request:
        assert calculate("10 + 20") == "30"
        request.assert_called_once()


def test_web_search_without_key(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "TAVILY_API_KEY" in web_search("latest AI news")
