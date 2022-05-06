from datetime import datetime

import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.internals.frame import DataFrame


def test_struct_various() -> None:
    df = pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    )
    s = df.to_struct("my_struct")

    assert s.struct.fields == ["int", "str", "bool", "list"]
    assert s[0] == {"int": 1, "str": "a", "bool": True, "list": [1, 2]}
    assert s[1] == {"int": 2, "str": "b", "bool": None, "list": [3]}
    assert s.struct.field("list").to_list() == [[1, 2], [3]]
    assert s.struct.field("int").to_list() == [1, 2]

    assert df.to_struct("my_struct").struct.to_frame().frame_equal(df)


def test_struct_to_list() -> None:
    assert pl.DataFrame(
        {"int": [1, 2], "str": ["a", "b"], "bool": [True, None], "list": [[1, 2], [3]]}
    ).select([pl.struct(pl.all()).alias("my_struct")]).to_series().to_list() == [
        {"int": 1, "str": "a", "bool": True, "list": [1, 2]},
        {"int": 2, "str": "b", "bool": None, "list": [3]},
    ]


def test_apply_to_struct() -> None:
    df = (
        pl.Series([None, 2, 3, 4])
        .apply(lambda x: {"a": x, "b": x * 2, "c": True, "d": [1, 2], "e": "foo"})
        .struct.to_frame()
    )

    expected = pl.DataFrame(
        {
            "a": [None, 2, 3, 4],
            "b": [None, 4, 6, 8],
            "c": [None, True, True, True],
            "d": [None, [1, 2], [1, 2], [1, 2]],
            "e": [None, "foo", "foo", "foo"],
        }
    )

    assert df.frame_equal(expected)


def test_rename_fields() -> None:
    df = pl.DataFrame({"int": [1, 2], "str": ["a", "b"], "bool": [True, None]})
    assert df.to_struct("my_struct").struct.rename_fields(["a", "b"]).struct.fields == [
        "a",
        "b",
    ]


def struct_unnesting() -> None:
    df = pl.DataFrame({"a": [1, 2]})
    out = df.select(
        [
            pl.all().alias("a_original"),
            pl.col("a")
            .apply(lambda x: (x, x * 2, x % 2 == 0))
            .struct.rename_fields(["a", "a_squared", "mod2eq0"])
            .alias("foo"),
        ]
    ).unnest("foo")

    expected = pl.DataFrame(
        {
            "a_original": [1, 2],
            "a": [1, 2],
            "a_squared": [2, 4],
            "mod2eq0": [False, True],
        }
    )

    assert out.frame_equal(expected)

    out = (
        df.lazy()
        .select(
            [
                pl.all().alias("a_original"),
                pl.col("a")
                .apply(lambda x: (x, x * 2, x % 2 == 0))
                .struct.rename_fields(["a", "a_squared", "mod2eq0"])
                .alias("foo"),
            ]
        )
        .unnest("foo")
        .collect()
    )
    out.frame_equal(expected)


def test_struct_function_expansion() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["one", "two", "three", "four"], "c": [9, 8, 7, 6]}
    )
    assert df.with_column(pl.struct(pl.col(["a", "b"])))["a"].struct.fields == [
        "a",
        "b",
    ]


def test_value_counts_expr() -> None:
    df = pl.DataFrame(
        {
            "id": ["a", "b", "b", "c", "c", "c"],
        }
    )

    out = (
        df.select(
            [
                pl.col("id").value_counts(),
            ]
        )
        .to_series()
        .to_list()
    )
    assert out == [
        {"id": "c", "counts": 3},
        {"id": "b", "counts": 2},
        {"id": "a", "counts": 1},
    ]


def test_struct_comparison() -> None:
    s1 = pl.DataFrame({"b": [1, 2, 3]}).to_struct("a")
    s2 = pl.DataFrame({"b": [0, 0, 0]}).to_struct("a")
    s3 = pl.DataFrame({"c": [1, 2, 3]}).to_struct("a")
    s4 = pl.DataFrame({"b": [1, 2, 3]}).to_struct("a")

    pl.testing.assert_series_equal(s1, s1)
    pl.testing.assert_series_equal(s1, s4)

    with pytest.raises(AssertionError):
        pl.testing.assert_series_equal(s1, s2)

    with pytest.raises(AssertionError):
        pl.testing.assert_series_equal(s1, s3)

    assert (s1 != s2).all() is True
    assert (s1 == s4).all() is True


def test_nested_struct() -> None:
    df = pl.DataFrame({"d": [1, 2, 3], "e": ["foo", "bar", "biz"]})
    # Nest the datafame
    nest_l1 = df.to_struct("c").to_frame()
    # Add another column on the same level
    nest_l1 = nest_l1.with_column(pl.col("c").is_nan().alias("b"))
    # Nest the dataframe again
    nest_l2 = nest_l1.to_struct("a").to_frame()

    assert isinstance(nest_l2.dtypes[0], pl.datatypes.Struct)
    assert nest_l2.dtypes[0].inner_types == nest_l1.dtypes
    assert isinstance(nest_l1.dtypes[0], pl.datatypes.Struct)


def test_eager_struct() -> None:
    s = pl.struct([pl.Series([1, 2, 3]), pl.Series(["a", "b", "c"])], eager=True)
    assert s.dtype == pl.Struct


def test_struct_to_pandas() -> None:
    df = pd.DataFrame([{"a": {"b": {"c": 2}}}])
    pl_df = pl.from_pandas(df)

    assert isinstance(pl_df.dtypes[0], pl.datatypes.Struct)

    assert pl_df.to_pandas().equals(df)


def test_struct_logical_types_to_pandas() -> None:
    timestamp = datetime(2022, 1, 1)
    df = pd.DataFrame([{"struct": {"timestamp": timestamp}}])
    assert pl.from_pandas(df).dtypes == [pl.Struct]


def test_struct_cols() -> None:
    """Test that struct columns can be imported and work as expected."""

    def build_struct_df(data: list) -> DataFrame:
        """Build Polars df from list of dicts. Can't import directly because of issue #3145."""
        arrow_df = pa.Table.from_pylist(data)
        polars_df = pl.from_arrow(arrow_df)
        assert isinstance(polars_df, DataFrame)
        return polars_df

    # struct column
    df = build_struct_df([{"struct_col": {"inner": 1}}])
    assert df.columns == ["struct_col"]
    assert df.schema == {"struct_col": pl.Struct}
    assert df["struct_col"].struct.field("inner").to_list() == [1]

    # struct in struct
    df = build_struct_df([{"nested_struct_col": {"struct_col": {"inner": 1}}}])
    assert df["nested_struct_col"].struct.field("struct_col").struct.field(
        "inner"
    ).to_list() == [1]

    # struct in list
    df = build_struct_df([{"list_of_struct_col": [{"inner": 1}]}])
    assert df["list_of_struct_col"][0].struct.field("inner").to_list() == [1]

    # struct in list in struct
    df = build_struct_df(
        [{"struct_list_struct_col": {"list_struct_col": [{"inner": 1}]}}]
    )
    assert df["struct_list_struct_col"].struct.field("list_struct_col")[0].struct.field(
        "inner"
    ).to_list() == [1]


def test_struct_with_validity() -> None:
    data = [{"a": {"b": 1}}, {"a": None}]
    tbl = pa.Table.from_pylist(data)
    df = pl.from_arrow(tbl)
    assert isinstance(df, pl.DataFrame)
    assert df["a"].to_list() == [{"b": 1}, {"b": None}]


def test_from_dicts_struct() -> None:
    assert pl.from_dicts([{"a": 1, "b": {"a": 1, "b": 2}}]).to_series(1).to_list() == [
        {"a": 1, "b": 2}
    ]

    assert pl.from_dicts(
        [{"a": 1, "b": {"a_deep": 1, "b_deep": {"a_deeper": [1, 2, 4]}}}]
    ).to_series(1).to_list() == [{"a_deep": 1, "b_deep": {"a_deeper": [1, 2, 4]}}]

    data = [
        {"a": [{"b": 0, "c": 1}]},
        {"a": [{"b": 1, "c": 2}]},
    ]

    assert pl.from_dicts(data).to_series().to_list() == [
        [{"b": 0, "c": 1}],
        [{"b": 1, "c": 2}],
    ]
