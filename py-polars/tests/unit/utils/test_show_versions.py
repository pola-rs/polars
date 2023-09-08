from typing import Any

import polars as pl


def test_show_versions(capsys: Any) -> None:
    pl.show_versions()

    out, _ = capsys.readouterr()
    assert "Python" in out
    assert "Polars" in out
    assert "Optional dependencies" in out
