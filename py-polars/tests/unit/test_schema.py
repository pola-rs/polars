import polars as pl


def test_schema() -> None:
    s = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})

    assert s["foo"] == pl.Int8()
    assert s["bar"] == pl.String()
    assert s.len() == 2
    assert s.names() == ["foo", "bar"]
    assert s.dtypes() == [pl.Int8(), pl.String()]


def test_schema_parse_nonpolars_dtypes() -> None:
    # Currently, no parsing is being done.
    s = pl.Schema({"foo": pl.List, "bar": int})  # type: ignore[arg-type]

    assert s["foo"] == pl.List
    assert s["bar"] == int
    assert s.len() == 2
    assert s.names() == ["foo", "bar"]
    assert s.dtypes() == [pl.List, int]


def test_schema_equality() -> None:
    s1 = pl.Schema({"foo": pl.Int8(), "bar": pl.Float64()})
    s2 = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})
    s3 = pl.Schema({"bar": pl.Float64(), "foo": pl.Int8()})
    assert s1 == s1
    assert s2 == s2
    assert s3 == s3
    assert s1 != s2
    assert s1 != s3
    assert s2 != s3
