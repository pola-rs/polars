import pickle
from datetime import datetime

import polars as pl


def test_schema() -> None:
    s = pl.Schema({"foo": pl.Int8(), "bar": pl.String()})

    assert s["foo"] == pl.Int8()
    assert s["bar"] == pl.String()
    assert s.len() == 2
    assert s.names() == ["foo", "bar"]
    assert s.dtypes() == [pl.Int8(), pl.String()]


def test_schema_parse_nonpolars_dtypes() -> None:
    cardinal_directions = pl.Enum(["north", "south", "east", "west"])

    s = pl.Schema({"foo": pl.List, "bar": int, "baz": cardinal_directions})  # type: ignore[arg-type]
    s["ham"] = datetime

    assert s["foo"] == pl.List
    assert s["bar"] == pl.Int64
    assert s["baz"] == cardinal_directions
    assert s["ham"] == pl.Datetime("us")

    assert s.len() == 4
    assert s.names() == ["foo", "bar", "baz", "ham"]
    assert s.dtypes() == [pl.List, pl.Int64, cardinal_directions, pl.Datetime("us")]

    assert list(s.to_python().values()) == [list, int, str, datetime]
    assert [tp.to_python() for tp in s.dtypes()] == [list, int, str, datetime]


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
    s = pl.Schema(
        {
            "foo": pl.Int8(),
            "bar": pl.String(),
            "ham": pl.Struct({"x": pl.List(pl.Date)}),
        }
    )
    pickled = pickle.dumps(s)
    s2 = pickle.loads(pickled)
    assert s == s2


def test_schema_python() -> None:
    input = {
        "foo": pl.Int8(),
        "bar": pl.String(),
        "baz": pl.Categorical("lexical"),
        "ham": pl.Object(),
        "spam": pl.Struct({"time": pl.List(pl.Duration), "dist": pl.Float64}),
    }
    expected = {
        "foo": int,
        "bar": str,
        "baz": str,
        "ham": object,
        "spam": dict,
    }
    for schema in (input, input.items(), list(input.items())):
        s = pl.Schema(schema)
        assert expected == s.to_python()


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
