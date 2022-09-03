from __future__ import annotations

import copy

import polars as pl


def test_copy() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": ["a", None], "c": [True, False]})
    assert copy.copy(df).frame_equal(df, True)
    assert copy.deepcopy(df).frame_equal(df, True)

    a = pl.Series("a", [1, 2])
    assert copy.copy(a).series_equal(a, True)
    assert copy.deepcopy(a).series_equal(a, True)


def test_categorical_round_trip() -> None:
    df = pl.DataFrame({"ints": [1, 2, 3], "cat": ["a", "b", "c"]})
    df = df.with_column(pl.col("cat").cast(pl.Categorical))

    tbl = df.to_arrow()
    assert "dictionary" in str(tbl["cat"].type)

    df2 = pl.from_arrow(tbl)
    assert df2.dtypes == [pl.Int64, pl.Categorical]


def test_date_list_fmt() -> None:
    df = pl.DataFrame(
        {
            "mydate": ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-05"],
            "index": [1, 2, 5, 5],
        }
    )

    df = df.with_column(pl.col("mydate").str.strptime(pl.Date, "%Y-%m-%d"))
    assert (
        str(df.groupby("index", maintain_order=True).agg(pl.col("mydate"))["mydate"])
        == """shape: (3,)
Series: 'mydate' [list]
[
	[2020-01-01]
	[2020-01-02]
	[2020-01-05, 2020-01-05]
]"""  # noqa: W191, E101
    )


def test_from_different_chunks() -> None:
    s0 = pl.Series("a", [1, 2, 3, 4, None])
    s1 = pl.Series("b", [1, 2])
    s11 = pl.Series("b", [1, 2, 3])
    s1.append(s11)

    # check we don't panic
    df = pl.DataFrame([s0, s1])
    df.to_arrow()
    df = pl.DataFrame([s0, s1])
    out = df.to_pandas()
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (5, 2)
