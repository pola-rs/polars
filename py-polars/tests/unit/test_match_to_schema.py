import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_match_to_schema() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int64),
            pl.Series("b", ["A", "B", "C"], pl.String),
        ]
    )

    result = df.lazy().match_to_schema(df.schema).collect()
    assert_frame_equal(df, result)

    result = df.lazy().match_to_schema({"a": pl.Int64(), "b": pl.String()}).collect()
    assert_frame_equal(df, result)

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema({"a": pl.String(), "b": pl.Int64()}).collect()

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema({"x": pl.Int64(), "y": pl.String()}).collect()


def test_match_to_schema_missing_columns() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int64),
            pl.Series("b", ["A", "B", "C"], pl.String),
        ]
    )

    expected = df.with_columns(c=pl.lit(None, dtype=pl.Datetime()))

    result = (
        df.lazy()
        .match_to_schema(
            expected.schema,
            missing_columns="insert",
        )
        .collect()
    )
    assert_frame_equal(expected, result)

    result = (
        df.lazy()
        .match_to_schema(
            expected.schema,
            missing_columns={"c": "insert"},
        )
        .collect()
    )

    result = (
        df.lazy()
        .match_to_schema(
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Datetime()},
            missing_columns="insert",
        )
        .collect()
    )
    assert_frame_equal(df.with_columns(c=pl.lit(None, dtype=pl.Datetime())), result)

    df = pl.DataFrame(
        [
            pl.Series("b", ["A", "B", "C"], pl.String),
        ]
    )

    result = (
        df.lazy()
        .match_to_schema(
            {"a": pl.Int64(), "b": pl.String(), "c": pl.Datetime()},
            missing_columns="insert",
        )
        .collect()
    )
    assert_frame_equal(
        df.select(
            a=pl.lit(None, dtype=pl.Int64()),
            b=pl.col.b,
            c=pl.lit(None, dtype=pl.Datetime()),
        ),
        result,
    )


def test_match_to_schema_extra_columns() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int64),
            pl.Series("b", ["A", "B", "C"], pl.String),
            pl.Series("c", [["A", "B"], ["C"], ["D"]], pl.List(pl.String)),
        ]
    )

    expected = df.select(["a"])

    with pytest.raises(pl.exceptions.SchemaError, match=r'`match_to_schema`: "b", "c"'):
        df.lazy().match_to_schema(expected.schema).collect()

    result = (
        df.lazy().match_to_schema(expected.schema, extra_columns="ignore").collect()
    )
    assert_frame_equal(expected, result)

    expected = df.select(["a", "b", "c"])
    result = (
        df.lazy().match_to_schema(expected.schema, extra_columns="ignore").collect()
    )
    assert_frame_equal(expected, result)

    expected = df.select(["a", "c"])
    result = (
        df.lazy().match_to_schema(expected.schema, extra_columns="ignore").collect()
    )
    assert_frame_equal(expected, result)


def test_match_to_schema_missing_struct_fields() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int64),
            pl.Series(
                "b", [{"x": "A"}, {"x": "B"}, {"x": "C"}], pl.Struct({"x": pl.String()})
            ),
            pl.Series("c", [["A", "B"], ["C"], ["D"]], pl.List(pl.String)),
        ]
    )

    expected = df.with_columns(
        pl.col.b.struct.with_fields(
            y=pl.repeat(pl.lit(None, dtype=pl.Datetime()), pl.len())
        )
    )

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema).collect()

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema, missing_columns="insert").collect()

    result = (
        df.lazy()
        .match_to_schema(expected.schema, missing_struct_fields="insert")
        .collect()
    )
    assert_frame_equal(expected, result)


def test_match_to_schema_extra_struct_fields() -> None:
    expected = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int64),
            pl.Series(
                "b", [{"x": "A"}, {"x": "B"}, {"x": "C"}], pl.Struct({"x": pl.String()})
            ),
            pl.Series("c", [["A", "B"], ["C"], ["D"]], pl.List(pl.String)),
        ]
    )

    df = expected.with_columns(
        pl.col.b.struct.with_fields(
            y=pl.repeat(pl.lit(None, dtype=pl.Datetime()), pl.len())
        )
    )

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema).collect()

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema, extra_columns="ignore").collect()

    result = (
        df.lazy()
        .match_to_schema(expected.schema, extra_struct_fields="ignore")
        .collect()
    )
    assert_frame_equal(expected, result)


def test_match_to_schema_int_upcast() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 2, 3], pl.Int32),
            pl.Series(
                "b", [{"x": "A"}, {"x": "B"}, {"x": "C"}], pl.Struct({"x": pl.String()})
            ),
            pl.Series("c", [["A", "B"], ["C"], ["D"]], pl.List(pl.String)),
        ]
    )

    expected = df.with_columns(pl.col.a.cast(pl.Int64))

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema).collect()

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema, float_cast="upcast").collect()

    result = df.lazy().match_to_schema(expected.schema, integer_cast="upcast").collect()
    assert_frame_equal(expected, result)


def test_match_to_schema_float_upcast() -> None:
    df = pl.DataFrame(
        [
            pl.Series("a", [1.0, 2.0, 3.0], pl.Float32()),
            pl.Series(
                "b", [{"x": "A"}, {"x": "B"}, {"x": "C"}], pl.Struct({"x": pl.String()})
            ),
            pl.Series("c", [["A", "B"], ["C"], ["D"]], pl.List(pl.String)),
        ]
    )

    expected = df.with_columns(pl.col.a.cast(pl.Float64()))

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema).collect()

    with pytest.raises(pl.exceptions.SchemaError):
        df.lazy().match_to_schema(expected.schema, integer_cast="upcast").collect()

    result = df.lazy().match_to_schema(expected.schema, float_cast="upcast").collect()
    assert_frame_equal(expected, result)
