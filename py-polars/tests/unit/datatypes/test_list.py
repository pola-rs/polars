from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import pytest

import polars as pl


def test_dtype() -> None:
    # inferred
    a = pl.Series("a", [[1, 2, 3], [2, 5], [6, 7, 8, 9]])
    assert a.dtype == pl.List
    assert a.inner_dtype == pl.Int64
    assert a.dtype.inner == pl.Int64  # type: ignore[union-attr]

    # explicit
    df = pl.DataFrame(
        data={
            "i": [[1, 2, 3]],
            "tm": [[time(10, 30, 45)]],
            "dt": [[date(2022, 12, 31)]],
            "dtm": [[datetime(2022, 12, 31, 1, 2, 3)]],
        },
        schema=[
            ("i", pl.List(pl.Int8)),
            ("tm", pl.List(pl.Time)),
            ("dt", pl.List(pl.Date)),
            ("dtm", pl.List(pl.Datetime)),
        ],
    )
    assert df.schema == {
        "i": pl.List(pl.Int8),
        "tm": pl.List(pl.Time),
        "dt": pl.List(pl.Date),
        "dtm": pl.List(pl.Datetime),
    }
    assert df.schema["i"].inner == pl.Int8  # type: ignore[union-attr]
    assert df.rows() == [
        (
            [1, 2, 3],
            [time(10, 30, 45)],
            [date(2022, 12, 31)],
            [datetime(2022, 12, 31, 1, 2, 3)],
        )
    ]


def test_categorical() -> None:
    # https://github.com/pola-rs/polars/issues/2038
    df = pl.DataFrame(
        [
            pl.Series("a", [1, 1, 1, 1, 1, 1, 1, 1]),
            pl.Series("b", [8, 2, 3, 6, 3, 6, 2, 2]),
            pl.Series("c", ["a", "b", "c", "a", "b", "c", "a", "b"]).cast(
                pl.Categorical
            ),
        ]
    )
    out = (
        df.groupby(["a", "b"])
        .agg(
            [
                pl.col("c").count().alias("num_different_c"),
                pl.col("c").alias("c_values"),
            ]
        )
        .filter(pl.col("num_different_c") >= 2)
        .to_series(3)
    )

    assert out.inner_dtype == pl.Categorical


def test_list_concat_rolling_window() -> None:
    # inspired by:
    # https://stackoverflow.com/questions/70377100/use-the-rolling-function-of-polars-to-get-a-list-of-all-values-in-the-rolling-wi
    # this tests if it works without specifically creating list dtype upfront. note that
    # the given answer is preferred over this snippet as that reuses the list array when
    # shifting
    df = pl.DataFrame(
        {
            "A": [1.0, 2.0, 9.0, 2.0, 13.0],
        }
    )
    out = df.with_columns(
        [pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)]
    ).select(
        [pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")]
    )
    assert out.shape == (5, 1)

    s = out.to_series()
    assert s.dtype == pl.List
    assert s.to_list() == [
        [None, None, 1.0],
        [None, 1.0, 2.0],
        [1.0, 2.0, 9.0],
        [2.0, 9.0, 2.0],
        [9.0, 2.0, 13.0],
    ]

    # this test proper null behavior of concat list
    out = (
        df.with_columns(pl.col("A").reshape((-1, 1)))  # first turn into a list
        .with_columns(
            [
                pl.col("A").shift(i).alias(f"A_lag_{i}")
                for i in range(3)  # slice the lists to a lag
            ]
        )
        .select(
            [
                pl.all(),
                pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias(
                    "A_rolling"
                ),
            ]
        )
    )
    assert out.shape == (5, 5)

    l64 = pl.List(pl.Float64)
    assert out.schema == {
        "A": l64,
        "A_lag_0": l64,
        "A_lag_1": l64,
        "A_lag_2": l64,
        "A_rolling": l64,
    }


def test_cast_inner() -> None:
    a = pl.Series([[1, 2]])
    for t in [bool, pl.Boolean]:
        b = a.cast(pl.List(t))
        assert b.dtype == pl.List(pl.Boolean)
        assert b.to_list() == [[True, True]]

    # this creates an inner null type
    df = pl.from_pandas(pd.DataFrame(data=[[[]], [[]]], columns=["A"]))
    assert (
        df["A"].cast(pl.List(int)).dtype.inner == pl.Int64  # type: ignore[union-attr]
    )


def test_list_empty_groupby_result_3521() -> None:
    # Create a left relation where the join column contains a null value
    left = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("groupby_column"),
            pl.lit(None).cast(pl.Int32).alias("join_column"),
        ]
    )

    # Create a right relation where there is a column to count distinct on
    right = pl.DataFrame().with_columns(
        [
            pl.lit(1).alias("join_column"),
            pl.lit(1).alias("n_unique_column"),
        ]
    )

    # Calculate n_unique after dropping nulls
    # This will panic on polars version 0.13.38 and 0.13.39
    assert (
        left.join(right, on="join_column", how="left")
        .groupby("groupby_column")
        .agg(pl.col("n_unique_column").drop_nulls())
    ).to_dict(False) == {"groupby_column": [1], "n_unique_column": [[]]}


def test_list_fill_null() -> None:
    df = pl.DataFrame({"C": [["a", "b", "c"], [], [], ["d", "e"]]})
    assert df.with_columns(
        [
            pl.when(pl.col("C").arr.lengths() == 0)
            .then(None)
            .otherwise(pl.col("C"))
            .alias("C")
        ]
    ).to_series().to_list() == [["a", "b", "c"], None, None, ["d", "e"]]


def test_list_fill_list() -> None:
    assert pl.DataFrame({"a": [[1, 2, 3], []]}).select(
        [
            pl.when(pl.col("a").arr.lengths() == 0)
            .then([5])
            .otherwise(pl.col("a"))
            .alias("filled")
        ]
    ).to_dict(False) == {"filled": [[1, 2, 3], [5]]}


def test_empty_list_construction() -> None:
    assert pl.Series([[]]).to_list() == [[]]
    assert pl.DataFrame([{"array": [], "not_array": 1234}], orient="row").to_dict(
        False
    ) == {"array": [[]], "not_array": [1234]}

    df = pl.DataFrame(schema=[("col", pl.List)])
    assert df.schema == {"col": pl.List}
    assert df.rows() == []


def test_list_concat_nulls() -> None:
    assert pl.DataFrame(
        {
            "a": [["a", "b"], None, ["c", "d", "e"], None],
            "t": [["x"], ["y"], None, None],
        }
    ).with_columns(pl.concat_list(["a", "t"]).alias("concat"))["concat"].to_list() == [
        ["a", "b", "x"],
        None,
        None,
        None,
    ]


def test_list_concat_supertype() -> None:
    df = pl.DataFrame(
        [pl.Series("a", [1, 2], pl.UInt8), pl.Series("b", [10000, 20000], pl.UInt16)]
    )
    assert df.with_columns(pl.concat_list(pl.col(["a", "b"])).alias("concat_list"))[
        "concat_list"
    ].to_list() == [[1, 10000], [2, 20000]]


def test_list_hash() -> None:
    out = pl.DataFrame({"a": [[1, 2, 3], [3, 4], [1, 2, 3]]}).with_columns(
        pl.col("a").hash().alias("b")
    )
    assert out.dtypes == [pl.List(pl.Int64), pl.UInt64]
    assert out[0, "b"] == out[2, "b"]


def test_list_diagonal_concat() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})

    df2 = pl.DataFrame({"b": [[1]]})

    assert pl.concat([df1, df2], how="diagonal").to_dict(False) == {
        "a": [1, 2, None],
        "b": [None, None, [1]],
    }


def test_is_in_empty_list_4559() -> None:
    assert pl.Series(["a"]).is_in([]).to_list() == [False]


def test_is_in_empty_list_4639() -> None:
    df = pl.DataFrame({"a": [1, None]})
    empty_list: list[int] = []

    assert df.with_columns([pl.col("a").is_in(empty_list).alias("a_in_list")]).to_dict(
        False
    ) == {"a": [1, None], "a_in_list": [False, False]}
    df = pl.DataFrame()
    assert df.with_columns(
        [pl.lit(None).cast(pl.Int64).is_in(empty_list).alias("in_empty_list")]
    ).to_dict(False) == {"in_empty_list": [False]}


def test_inner_type_categorical_on_rechunk() -> None:
    df = pl.DataFrame({"cats": ["foo", "bar"]}).select(
        pl.col(pl.Utf8).cast(pl.Categorical).list()
    )

    assert pl.concat([df, df], rechunk=True).dtypes == [pl.List(pl.Categorical)]


def test_groupby_list_column() -> None:
    df = (
        pl.DataFrame({"a": ["a", "b", "a"]})
        .with_columns(pl.col("a").cast(pl.Categorical))
        .groupby("a", maintain_order=True)
        .agg(pl.col("a").alias("a_list"))
    )

    assert df.groupby("a_list", maintain_order=True).first().to_dict(False) == {
        "a_list": [["a", "a"], ["b"]],
        "a": ["a", "b"],
    }


def test_fast_explode_flag() -> None:
    df1 = pl.DataFrame({"values": [[[1, 2]]]})
    assert df1.clone().vstack(df1)["values"].flags["FAST_EXPLODE"]

    # test take that produces a null in list
    df = pl.DataFrame({"a": [1, 2, 1, 3]})
    df_b = pl.DataFrame({"a": [1, 2], "c": [["1", "2", "c"], ["1", "2", "c"]]})
    assert df_b["c"].flags["FAST_EXPLODE"]

    # join produces a null
    assert not (df.join(df_b, on=["a"], how="left").select(["c"]))["c"].flags[
        "FAST_EXPLODE"
    ]


def test_fast_explode_on_list_struct_6208() -> None:
    data = [
        {
            "label": "l",
            "tag": "t",
            "ref": 1,
            "parents": [{"ref": 1, "tag": "t", "ratio": 62.3}],
        },
        {"label": "l", "tag": "t", "ref": 1, "parents": None},
    ]

    df = pl.DataFrame(
        data,
        schema={
            "label": pl.Utf8,
            "tag": pl.Utf8,
            "ref": pl.Int64,
            "parents": pl.List(
                pl.Struct({"ref": pl.Int64, "tag": pl.Utf8, "ratio": pl.Float64})
            ),
        },
    )

    assert not df["parents"].flags["FAST_EXPLODE"]
    assert df.explode("parents").to_dict(False) == {
        "label": ["l", "l"],
        "tag": ["t", "t"],
        "ref": [1, 1],
        "parents": [
            {"ref": 1, "tag": "t", "ratio": 62.3},
            {"ref": None, "tag": None, "ratio": None},
        ],
    }


def test_concat_list_in_agg_6397() -> None:
    df = pl.DataFrame({"group": [1, 2, 2, 3], "value": ["a", "b", "c", "d"]})

    # single list
    assert df.groupby("group").agg(
        [
            # this casts every element to a list
            pl.concat_list(pl.col("value")),
        ]
    ).sort("group").to_dict(False) == {
        "group": [1, 2, 3],
        "value": [[["a"]], [["b"], ["c"]], [["d"]]],
    }

    # nested list
    assert df.groupby("group").agg(
        [
            pl.concat_list(pl.col("value").list()).alias("result"),
        ]
    ).sort("group").to_dict(False) == {
        "group": [1, 2, 3],
        "result": [[["a"]], [["b", "c"]], [["d"]]],
    }


def test_concat_list_empty_raises() -> None:
    with pytest.raises(pl.ComputeError):
        pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.concat_list([]))


def test_flat_aggregation_to_list_conversion_6918() -> None:
    df = pl.DataFrame({"a": [1, 2, 2], "b": [[0, 1], [2, 3], [4, 5]]})

    assert df.groupby("a", maintain_order=True).agg(
        pl.concat_list([pl.col("b").arr.get(i).mean().list() for i in range(2)])
    ).to_dict(False) == {"a": [1, 2], "b": [[[0.0, 1.0]], [[3.0, 4.0]]]}


def test_concat_list_with_lit() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})

    assert df.select(pl.concat_list([pl.col("a"), pl.lit(1)]).alias("a")).to_dict(
        False
    ) == {"a": [[1, 1], [2, 1], [3, 1]]}

    assert df.select(pl.concat_list([pl.lit(1), pl.col("a")]).alias("a")).to_dict(
        False
    ) == {"a": [[1, 1], [1, 2], [1, 3]]}


def test_list_count_match() -> None:
    assert pl.DataFrame({"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}).select(
        pl.col("listcol").arr.count_match(2).alias("number_of_twos")
    ).to_dict(False) == {"number_of_twos": [0, 0, 2, 1, 0]}
    assert pl.DataFrame({"listcol": [[], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]]}).select(
        pl.col("listcol").arr.count_match(2).alias("number_of_twos")
    ).to_dict(False) == {"number_of_twos": [0, 0, 2, 1, 0]}


def test_list_sum_and_dtypes() -> None:
    # ensure the dtypes of sum align with normal sum
    for dt_in, dt_out in [
        (pl.Int8, pl.Int64),
        (pl.Int16, pl.Int64),
        (pl.Int32, pl.Int32),
        (pl.Int64, pl.Int64),
        (pl.UInt8, pl.Int64),
        (pl.UInt16, pl.Int64),
        (pl.UInt32, pl.UInt32),
        (pl.UInt64, pl.UInt64),
    ]:
        df = pl.DataFrame(
            {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]},
            schema={"a": pl.List(dt_in)},
        )

        summed = df.explode("a").sum()
        assert summed.dtypes == [dt_out]
        assert summed.item() == 32
        assert df.select(pl.col("a").arr.sum()).dtypes == [dt_out]

    assert df.select(pl.col("a").arr.sum()).to_dict(False) == {"a": [1, 6, 10, 15]}

    # include nulls
    assert pl.DataFrame(
        {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]}
    ).select(pl.col("a").arr.sum()).to_dict(False) == {"a": [1, 6, 10, 15, None]}


def test_list_mean() -> None:
    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]}).select(
        pl.col("a").arr.mean()
    ).to_dict(False) == {"a": [1.0, 2.0, 2.5, 3.0]}

    assert pl.DataFrame({"a": [[1], [1, 2, 3], [1, 2, 3, 4], None]}).select(
        pl.col("a").arr.mean()
    ).to_dict(False) == {"a": [1.0, 2.0, 2.5, None]}


def test_list_min_max() -> None:
    for dt in pl.NUMERIC_DTYPES:
        if dt == pl.Decimal:
            continue
        df = pl.DataFrame(
            {"a": [[1], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]},
            schema={"a": pl.List(dt)},
        )
        assert df.select(pl.col("a").arr.min())["a"].series_equal(
            df.select(pl.col("a").arr.first())["a"]
        )
        assert df.select(pl.col("a").arr.max())["a"].series_equal(
            df.select(pl.col("a").arr.last())["a"]
        )

    df = pl.DataFrame(
        {"a": [[1], [1, 5, -1, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5], None]},
    )
    assert df.select(pl.col("a").arr.min()).to_dict(False) == {"a": [1, -1, 1, 1, None]}
    assert df.select(pl.col("a").arr.max()).to_dict(False) == {"a": [1, 5, 4, 5, None]}
