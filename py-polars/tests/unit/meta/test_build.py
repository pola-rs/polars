import polars as pl


def test_build_info_version() -> None:
    build_info = pl.build_info()
    assert build_info["version"] == pl.__version__
