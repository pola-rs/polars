import polars as pl


def test_build_info_version() -> None:
    build_info = pl.build_info()
    assert build_info["version"] == pl.__version__


def test_build_info_keys() -> None:
    build_info = pl.build_info()
    expected_keys = [
        "compiler",
        "time",
        "dependencies",
        "features",
        "host",
        "target",
        "git",
        "version",
    ]
    assert sorted(build_info.keys()) == sorted(expected_keys)


def test_build_info_features() -> None:
    build_info = pl.build_info()
    assert "BUILD_INFO" in build_info["features"]
