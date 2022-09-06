from __future__ import annotations

import io
import typing
from datetime import date

import numpy as np
import pytest

import polars as pl


def test_error_on_empty_groupby() -> None:
    with pytest.raises(
        pl.ComputeError, match="expected keys in groupby operation, got nothing"
    ):
        pl.DataFrame({"x": [0, 0, 1, 1]}).groupby([]).agg(pl.count())


def test_error_on_reducing_map() -> None:
    df = pl.DataFrame(
        {"id": [0, 0, 0, 1, 1, 1], "t": [2, 4, 5, 10, 11, 14], "y": [0, 1, 1, 2, 3, 4]}
    )

    with pytest.raises(
        pl.ComputeError,
        match=(
            "A 'map' functions output length must be equal to that of the input length."
            " Consider using 'apply' in favor of 'map'."
        ),
    ):
        df.groupby("id").agg(pl.map(["t", "y"], np.trapz))


def test_error_on_invalid_by_in_asof_join() -> None:
    df1 = pl.DataFrame(
        {
            "a": ["a", "b", "a"],
            "b": [1, 2, 3],
            "c": ["a", "b", "a"],
        }
    )

    df2 = df1.with_column(pl.col("a").cast(pl.Categorical))
    with pytest.raises(pl.ComputeError):
        df1.join_asof(df2, on="b", by=["a", "c"])


def test_not_found_error() -> None:
    csv = "a,b,c\n2,1,1"
    df = pl.read_csv(io.StringIO(csv))
    with pytest.raises(pl.NotFoundError):
        df.select("d")


def test_string_numeric_comp_err() -> None:
    with pytest.raises(pl.ComputeError, match="cannot compare Utf8 with numeric data"):
        pl.DataFrame({"a": [1.1, 21, 31, 21, 51, 61, 71, 81]}).select(pl.col("a") < "9")


def test_panic_exception() -> None:
    with pytest.raises(
        pl.PanicException,
        match=r"""this operation is not implemented/valid for this dtype: .*""",
    ):
        pl.struct(pl.Series("a", [1, 2, 3]), eager=True).sort()


@typing.no_type_check
def test_join_lazy_on_df() -> None:
    df_left = pl.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Names": ["A", "B", "C", "D"],
        }
    )
    df_right = pl.DataFrame({"Id": [1, 3], "Tags": ["xxx", "yyy"]})

    with pytest.raises(
        ValueError,
        match=(
            "Expected a `LazyFrame` as join table, got"
            " <class 'polars.internals.dataframe.frame.DataFrame'>"
        ),
    ):
        df_left.lazy().join(df_right, on="Id")

    with pytest.raises(
        ValueError,
        match=(
            "Expected a `LazyFrame` as join table, got"
            " <class 'polars.internals.dataframe.frame.DataFrame'>"
        ),
    ):
        df_left.lazy().join_asof(df_right, on="Id")


def test_projection_update_schema_missing_column() -> None:
    with pytest.raises(
        pl.ComputeError, match="column 'colC' not available in schema Schema:*"
    ):
        (
            pl.DataFrame({"colA": ["a", "b", "c"], "colB": [1, 2, 3]})
            .lazy()
            .filter(~pl.col("colC").is_null())
            .groupby(["colA"])
            .agg([pl.col("colB").sum().alias("result")])
            .collect()
        )


def test_not_found_on_rename() -> None:
    df = pl.DataFrame({"exists": [1, 2, 3]})

    with pytest.raises(pl.NotFoundError):
        df.rename(
            {
                "does_not_exist": "exists",
            }
        )


@typing.no_type_check
@pytest.mark.filterwarnings("ignore:setting a DataFrame by indexing:DeprecationWarning")
def test_getitem_errs() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        ValueError,
        match=r"Cannot __getitem__ on DataFrame with item: "
        r"'{'some'}' of type: '<class 'set'>'.",
    ):
        df[{"some"}]

    with pytest.raises(
        ValueError,
        match=r"Cannot __getitem__ on Series of dtype: "
        r"'<class 'polars.datatypes.Int64'>' with argument: "
        r"'{'strange'}' of type: '<class 'set'>'.",
    ):
        df["a"][{"strange"}]

    with pytest.raises(
        ValueError,
        match=r"Cannot __setitem__ on "
        r"DataFrame with key: '{'some'}' of "
        r"type: '<class 'set'>' and value: "
        r"'foo' of type: '<class 'str'>'",
    ):
        df[{"some"}] = "foo"


def test_err_bubbling_up_to_lit() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "value": [42]})

    with pytest.raises(ValueError):
        df.filter(pl.col("date") == pl.Date("2020-01-01"))


def test_error_on_double_agg() -> None:
    for e in [
        "mean",
        "max",
        "min",
        "sum",
        "std",
        "var",
        "n_unique",
        "last",
        "first",
        "median",
        "skew",  # this one is comes from Apply
    ]:
        with pytest.raises(pl.ComputeError, match="The column is already aggregated"):
            (
                pl.DataFrame(
                    {
                        "a": [1, 1, 1, 2, 2],
                        "b": [1, 2, 3, 4, 5],
                    }
                )
                .groupby("a")
                .agg([getattr(pl.col("b").min(), e)()])
            )
