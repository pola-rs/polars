import polars as pl


def test_show_versions(capsys) -> None:  # type: ignore[no-untyped-def]
    pl.show_versions()

    out, _ = capsys.readouterr()
    assert "Python" in out
    assert "Polars" in out
    assert "Optional dependencies" in out


def test_build_info() -> None:
    build_info = pl.build_info()
    features = build_info.get("features", {})
    if features:
        assert "BUILD_INFO" in features
    else:
        assert "version" in build_info
