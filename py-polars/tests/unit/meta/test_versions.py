from typing import Any

import pytest

import polars as pl


@pytest.mark.slow()
def test_show_versions(capsys: Any) -> None:
    pl.show_versions()

    out, _ = capsys.readouterr()
    assert "Python" in out
    assert "Polars" in out
    assert "Optional dependencies" in out
