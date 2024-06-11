import polars as pl


def test_schema() -> None:
    s = pl.Schema({"a": pl.Int8(), "b": pl.String()})

    assert s["a"] == pl.Int8()
    assert s["b"] == pl.String()
    assert s.len() == 2
    assert s.names() == ["a", "b"]
    assert s.dtypes() == [pl.Int8(), pl.String()]


def test_schema_parse_nonpolars_dtypes() -> None:
    # Currently, no parsing is being done.
    s = pl.Schema({"a": pl.List, "b": int})  # type: ignore[arg-type]

    assert s["a"] == pl.List
    assert s["b"] == int
    assert s.len() == 2
    assert s.names() == ["a", "b"]
    assert s.dtypes() == [pl.List, int]


def test_schema_equality() -> None:
    s1 = pl.Schema({"a": pl.Int8(), "b": pl.Float64()})
    s2 = pl.Schema({"b": pl.Float64(), "a": pl.Int8()})
    assert s1 == s1
    assert s2 == s2
    assert s1 != s2
