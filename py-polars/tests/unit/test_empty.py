import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_empty_str_concat_lit() -> None:
    df = pl.DataFrame({"a": [], "b": []}, schema=[("a", pl.String), ("b", pl.String)])
    assert df.with_columns(pl.lit("asd") + pl.col("a")).schema == {
        "a": pl.String,
        "b": pl.String,
        "literal": pl.String,
    }


def test_empty_cross_join() -> None:
    a = pl.LazyFrame(schema={"a": pl.Int32})
    b = pl.LazyFrame(schema={"b": pl.Int32})

    assert (a.join(b, how="cross").collect()).schema == {"a": pl.Int32, "b": pl.Int32}


def test_empty_string_replace() -> None:
    s = pl.Series("", [], dtype=pl.String)
    assert_series_equal(s.str.replace("a", "b", literal=True), s)
    assert_series_equal(s.str.replace("a", "b"), s)
    assert_series_equal(s.str.replace("ab", "b", literal=True), s)
    assert_series_equal(s.str.replace("ab", "b"), s)


def test_empty_window_function() -> None:
    expr = (pl.col("VAL") / pl.col("VAL").sum()).over("KEY")

    df = pl.DataFrame(schema={"KEY": pl.String, "VAL": pl.Float64})
    df.select(expr)  # ComputeError

    lf = pl.DataFrame(schema={"KEY": pl.String, "VAL": pl.Float64}).lazy()
    expected = pl.DataFrame(schema={"VAL": pl.Float64})
    assert_frame_equal(lf.select(expr).collect(), expected)


def test_empty_count_window() -> None:
    df = pl.DataFrame(
        {"ID": [], "DESC": [], "dataset": []},
        schema={"ID": pl.String, "DESC": pl.String, "dataset": pl.String},
    )

    out = df.select(pl.col("ID").count().over(["ID", "DESC"]))
    assert out.schema == {"ID": pl.UInt32}
    assert out.height == 0


def test_empty_sort_by_args() -> None:
    df = pl.DataFrame([1, 2, 3])
    with pytest.raises(InvalidOperationError):
        df.select(pl.all().sort_by([]))


def test_empty_9137() -> None:
    out = (
        pl.DataFrame(
            {"id": [], "value": []},
            schema={"id": pl.Float32, "value": pl.Float32},
        )
        .group_by("id")
        .agg(pl.col("value").pow(2).mean())
    )
    assert out.shape == (0, 2)
    assert out.dtypes == [pl.Float32, pl.Float32]


@pytest.mark.parametrize("dtype", [pl.String, pl.Binary, pl.UInt32])
@pytest.mark.parametrize(
    "set_operation",
    ["set_intersection", "set_union", "set_difference", "set_symmetric_difference"],
)
def test_empty_df_set_operations(set_operation: str, dtype: pl.DataType) -> None:
    expr = getattr(pl.col("list1").list, set_operation)(pl.col("list2"))
    df = pl.DataFrame([], {"list1": pl.List(dtype), "list2": pl.List(dtype)})
    assert df.select(expr).is_empty()


def test_empty_set_intersection() -> None:
    full = pl.Series("full", [[1, 2, 3]], pl.List(pl.UInt32))
    empty = pl.Series("empty", [[]], pl.List(pl.UInt32))

    assert_series_equal(empty.rename("full"), full.list.set_intersection(empty))
    assert_series_equal(empty, empty.list.set_intersection(full))


def test_empty_set_difference() -> None:
    full = pl.Series("full", [[1, 2, 3]], pl.List(pl.UInt32))
    empty = pl.Series("empty", [[]], pl.List(pl.UInt32))

    assert_series_equal(full, full.list.set_difference(empty))
    assert_series_equal(empty, empty.list.set_difference(full))


def test_empty_set_union() -> None:
    full = pl.Series("full", [[1, 2, 3]], pl.List(pl.UInt32))
    empty = pl.Series("empty", [[]], pl.List(pl.UInt32))

    assert_series_equal(full, full.list.set_union(empty))
    assert_series_equal(full.rename("empty"), empty.list.set_union(full))


def test_empty_set_symmetric_difference() -> None:
    full = pl.Series("full", [[1, 2, 3]], pl.List(pl.UInt32))
    empty = pl.Series("empty", [[]], pl.List(pl.UInt32))

    assert_series_equal(full, full.list.set_symmetric_difference(empty))
    assert_series_equal(full.rename("empty"), empty.list.set_symmetric_difference(full))


@pytest.mark.parametrize("name", ["sort", "unique", "head", "tail", "shift", "reverse"])
def test_empty_list_namespace_output_9585(name: str) -> None:
    dtype = pl.List(pl.String)
    df = pl.DataFrame([[None]], schema={"A": dtype})

    expr = getattr(pl.col("A").list, name)()
    result = df.select(expr)

    assert result.dtypes == df.dtypes


def test_empty_is_in() -> None:
    assert_series_equal(
        pl.Series("a", [1, 2, 3]).is_in([]), pl.Series("a", [False] * 3)
    )


@pytest.mark.parametrize("method", ["drop_nulls", "unique"])
def test_empty_to_empty(method: str) -> None:
    assert getattr(pl.DataFrame(), method)().shape == (0, 0)


def test_empty_shift_over_16676() -> None:
    df = pl.DataFrame({"a": [], "b": []})
    assert df.with_columns(pl.col("a").shift(fill_value=0).over("b")).shape == (0, 2)


def test_empty_list_cat_16405() -> None:
    df = pl.DataFrame(schema={"cat": pl.List(pl.Categorical)})
    df.select(pl.col("cat") == pl.col("cat"))


def test_empty_list_concat_16924() -> None:
    df = pl.DataFrame(schema={"a": pl.Int16, "b": pl.List(pl.String)})
    df.with_columns(pl.col("b").list.concat([pl.col("a").cast(pl.String)]))


def test_empty_input_expansion() -> None:
    df = pl.DataFrame({"A": [1], "B": [2]})

    with pytest.raises(pl.exceptions.InvalidOperationError):
        (
            df.select("A", "B").with_columns(
                pl.col("B").sort_by(pl.struct(pl.exclude("A", "B")))
            )
        )
