import polars as pl


def test_build_info() -> None:
    build_info = pl.build_info()
    assert "version" in build_info  # version is always present
    features = build_info.get("features", {})
    if features:  # only when compiled with `build_info` feature gate
        assert "BUILD_INFO" in features
