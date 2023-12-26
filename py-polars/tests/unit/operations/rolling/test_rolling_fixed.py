import polars as pl


def test_rolling_var_stability_12905() -> None:
    s1 = pl.Series("a", [36743.6 for _ in range(10)])
    assert s1.rolling_var(window_size=12, min_periods=2).sum() == 0.0
    assert s1.rolling_std(window_size=12, min_periods=2).sum() == 0.0
