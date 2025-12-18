from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_map_rows() -> None:
    df = pl.DataFrame({"a": ["foo", "2"], "b": [1, 2], "c": [1.0, 2.0]})

    result = df.map_rows(lambda x: len(x), None)

    expected = pl.DataFrame({"map": [3, 3]})
    assert_frame_equal(result, expected)


def test_map_rows_list_return() -> None:
    df = pl.DataFrame({"start": [1, 2], "end": [3, 5]})

    result = df.map_rows(lambda r: pl.Series(range(r[0], r[1] + 1)))

    expected = pl.DataFrame({"map": [[1, 2, 3], [2, 3, 4, 5]]})
    assert_frame_equal(result, expected)


def test_map_rows_dataframe_return() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["c", "d", None]})

    result = df.map_rows(lambda row: (row[0] * 10, "foo", True, row[-1]))

    expected = pl.DataFrame(
        {
            "column_0": [10, 20, 30],
            "column_1": ["foo", "foo", "foo"],
            "column_2": [True, True, True],
            "column_3": ["c", "d", None],
        }
    )
    assert_frame_equal(result, expected)


def test_map_rows_shifted_chunks() -> None:
    df = pl.DataFrame(pl.Series("texts", ["test", "test123", "tests"]))
    df = df.select(pl.col("texts"), pl.col("texts").shift(1).alias("texts_shifted"))

    result = df.map_rows(lambda x: x)

    expected = pl.DataFrame(
        {
            "column_0": ["test", "test123", "tests"],
            "column_1": [None, "test", "test123"],
        }
    )
    assert_frame_equal(result, expected)


def test_map_elements_infer() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 2, 3],
        }
    )
    lf = lf.select(pl.col.a.map_elements(lambda v: f"pre-{v}"))

    # this should not go through execution, solely through the planner
    schema = lf.collect_schema()

    assert schema.names() == ["a"]
    assert schema.dtypes() == [pl.String]


def test_map_rows_object_dtype() -> None:
    df = pl.DataFrame(
        {
            "a": [0, 0, 1, 2, 2],
            "b": [object(), 2, 0, 0, 1],
        },
        schema={"a": pl.Int64, "b": pl.Object},
    )

    out = df.map_rows(lambda _d: 1)
    assert_frame_equal(out, pl.DataFrame({"map": [1, 1, 1, 1, 1]}))
