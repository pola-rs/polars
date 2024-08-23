import pickle

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
    assert s["bar"] is int
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


def test_schema_picklable() -> None:
    s = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})

    pickled = pickle.dumps(s)
    s2 = pickle.loads(pickled)

    assert s == s2


def test_schema_in_map_elements_returns_scalar() -> None:
    schema = pl.Schema([("portfolio", pl.String()), ("irr", pl.Float64())])

    ldf = pl.LazyFrame(
        {
            "portfolio": ["A", "A", "B", "B"],
            "amounts": [100.0, -110.0] * 2,
        }
    )

    q = ldf.group_by("portfolio").agg(
        pl.col("amounts")
        .map_elements(
            lambda x: float(x.sum()), return_dtype=pl.Float64, returns_scalar=True
        )
        .alias("irr")
    )

    assert (q.collect_schema()) == schema
    assert q.collect().schema == schema


def test_ir_cache_unique_18198() -> None:
    lf = pl.LazyFrame({"a": [1]})
    lf.collect_schema()
    assert pl.concat([lf, lf]).collect().to_dict(as_series=False) == {"a": [1, 1]}
