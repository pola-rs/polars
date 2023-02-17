import polars as pl


def test_show_versions(capsys) -> None:  # type: ignore[no-untyped-def]
    pl.show_versions()

    out, _ = capsys.readouterr()
    assert "Python" in out
    assert "Polars" in out
    assert "Optional dependencies" in out
