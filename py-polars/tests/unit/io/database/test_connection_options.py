"""Tests for the connection_options parameter in read_database_uri."""

from __future__ import annotations

from types import ModuleType
from typing import Any, NoReturn
from urllib.parse import parse_qs, urlparse

import pytest

import polars as pl


@pytest.fixture
def _mock_connectorx(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a fake connectorx module that captures the connection URI."""
    fake_cx = ModuleType("connectorx")
    fake_cx.__version__ = "0.4.6"  # type: ignore[attr-defined]

    def _read_sql(**kwargs: Any) -> NoReturn:
        raise ConnectionError(kwargs["conn"])

    fake_cx.read_sql = _read_sql  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "connectorx", fake_cx)


@pytest.mark.usefixtures("_mock_connectorx")
class TestConnectionOptions:
    """Verify that connection_options dict gets merged into the URI."""

    def _get_uri(self, uri: str, **kwargs: Any) -> str:
        """Run read_database_uri and extract the URI from the raised error."""
        with pytest.raises(ConnectionError) as exc_info:
            pl.read_database_uri("SELECT 1", uri, engine="connectorx", **kwargs)
        return str(exc_info.value)

    def _get_params(self, uri: str, **kwargs: Any) -> dict[str, list[str]]:
        """Extract parsed query parameters from the resulting URI."""
        return parse_qs(urlparse(self._get_uri(uri, **kwargs)).query)

    def test_adds_params_to_uri(self) -> None:
        params = self._get_params(
            "trino://user@host:8080/catalog",
            connection_options={"schema": "analytics", "source": "test"},
        )
        assert params["schema"] == ["analytics"]
        assert params["source"] == ["test"]

    def test_preserves_existing_params(self) -> None:
        params = self._get_params(
            "trino://user@host:8080/catalog?verify=false",
            connection_options={"source": "test"},
        )
        assert params["verify"] == ["false"]
        assert params["source"] == ["test"]

    def test_overrides_duplicate_params(self) -> None:
        params = self._get_params(
            "trino://user@host:8080/catalog?schema=old",
            connection_options={"schema": "new"},
        )
        assert params["schema"] == ["new"]

    def test_non_string_values(self) -> None:
        params = self._get_params(
            "trino://user@host:8080/catalog",
            connection_options={"SSL": True, "timeout": 30},
        )
        assert params["SSL"] == ["True"]
        assert params["timeout"] == ["30"]

    def test_url_encodes_values(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options={"client_info": "my app v2"},
        )
        assert "my%20app%20v2" in result

    def test_none_is_noop(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options=None,
        )
        assert result == "trino://user@host:8080/catalog"

    def test_empty_dict_is_noop(self) -> None:
        result = self._get_uri(
            "trino://user@host:8080/catalog",
            connection_options={},
        )
        assert result == "trino://user@host:8080/catalog"

    def test_rejected_for_adbc(self) -> None:
        with pytest.raises(
            ValueError, match=r"adbc.*does not support.*connection_options"
        ):
            pl.read_database_uri(
                "SELECT 1",
                "sqlite:///:memory:",
                engine="adbc",
                connection_options={"key": "value"},
            )
