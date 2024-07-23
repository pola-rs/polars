from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

import polars as pl
from polars.testing import assert_series_equal


def is_sorted_any(s: pl.Series) -> bool:
    return s.flags["SORTED_ASC"] or s.flags["SORTED_DESC"]


def is_not_sorted(s: pl.Series) -> bool:
    return not is_sorted_any(s)


def test_sorted_flag_14552() -> None:
    a = pl.DataFrame({"a": [2, 1, 3]})

    a = pl.concat([a, a], rechunk=False)
    assert not a.join(a, on="a", how="left")["a"].flags["SORTED_ASC"]


def test_sorted_flag_concat_15072() -> None:
    # Both all-null
    a = pl.Series("x", [None, None], dtype=pl.Int8)
    b = pl.Series("x", [None, None], dtype=pl.Int8)
    assert pl.concat((a, b)).flags["SORTED_ASC"]

    # left all-null, right 0 < null_count < len
    a = pl.Series("x", [None, None], dtype=pl.Int8)
    b = pl.Series("x", [1, 2, 1, None], dtype=pl.Int8)

    out = pl.concat((a, b.sort()))
    assert out.to_list() == [None, None, None, 1, 1, 2]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((a, b.sort(descending=True)))
    assert out.to_list() == [None, None, None, 2, 1, 1]
    assert out.flags["SORTED_DESC"]

    out = pl.concat((a, b.sort(nulls_last=True)))
    assert out.to_list() == [None, None, 1, 1, 2, None]
    assert is_not_sorted(out)

    out = pl.concat((a, b.sort(nulls_last=True, descending=True)))
    assert out.to_list() == [None, None, 2, 1, 1, None]
    assert is_not_sorted(out)

    # left 0 < null_count < len, right all-null
    a = pl.Series("x", [1, 2, 1, None], dtype=pl.Int8)
    b = pl.Series("x", [None, None], dtype=pl.Int8)

    out = pl.concat((a.sort(), b))
    assert out.to_list() == [None, 1, 1, 2, None, None]
    assert is_not_sorted(out)

    out = pl.concat((a.sort(descending=True), b))
    assert out.to_list() == [None, 2, 1, 1, None, None]
    assert is_not_sorted(out)

    out = pl.concat((a.sort(nulls_last=True), b))
    assert out.to_list() == [1, 1, 2, None, None, None]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((a.sort(nulls_last=True, descending=True), b))
    assert out.to_list() == [2, 1, 1, None, None, None]
    assert out.flags["SORTED_DESC"]

    # both 0 < null_count < len
    assert pl.concat(
        (
            pl.Series([None, 1]).set_sorted(),
            pl.Series([2]).set_sorted(),
        )
    ).flags["SORTED_ASC"]

    assert is_not_sorted(
        pl.concat(
            (
                pl.Series([None, 1]).set_sorted(),
                pl.Series([2, None]).set_sorted(),
            )
        )
    )

    assert pl.concat(
        (
            pl.Series([None, 2]).set_sorted(descending=True),
            pl.Series([1]).set_sorted(descending=True),
        )
    ).flags["SORTED_DESC"]

    assert is_not_sorted(
        pl.concat(
            (
                pl.Series([None, 2]).set_sorted(descending=True),
                pl.Series([1, None]).set_sorted(descending=True),
            )
        )
    )

    # Concat with empty series
    s = pl.Series([None, 1]).set_sorted()

    out = pl.concat((s.clear(), s))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    out = pl.concat((s, s.clear()))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    s = pl.Series([1, None]).set_sorted()

    out = pl.concat((s.clear(), s))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]

    out = pl.concat((s, s.clear()))
    assert_series_equal(out, s)
    assert out.flags["SORTED_ASC"]


@pytest.mark.parametrize("unit_descending", [True, False])
def test_sorted_flag_concat_unit(unit_descending: bool) -> None:
    unit = pl.Series([1]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [1, 2, 3]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, 1]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [1, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, 1]
    assert out.flags["SORTED_DESC"]

    # unit with nulls first
    unit = pl.Series([None, 1]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [None, 1, 2, 3]
    assert out.flags["SORTED_ASC"]

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, None, 1]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [None, 1, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, None, 1]
    assert is_not_sorted(out)

    # unit with nulls last
    unit = pl.Series([1, None]).set_sorted(descending=unit_descending)

    a = unit
    b = pl.Series([2, 3]).set_sorted()

    out = pl.concat((a, b))
    assert out.to_list() == [1, None, 2, 3]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [2, 3, 1, None]
    assert is_not_sorted(out)

    a = unit
    b = pl.Series([3, 2]).set_sorted(descending=True)

    out = pl.concat((a, b))
    assert out.to_list() == [1, None, 3, 2]
    assert is_not_sorted(out)

    out = pl.concat((b, a))
    assert out.to_list() == [3, 2, 1, None]
    assert out.flags["SORTED_DESC"]


def test_sorted_flag_null() -> None:
    assert pl.DataFrame({"x": [None] * 2})["x"].flags["SORTED_ASC"] is False


def test_sorted_update_flags_10327() -> None:
    assert pl.concat(
        [
            pl.Series("a", [1], dtype=pl.Int64).to_frame(),
            pl.Series("a", [], dtype=pl.Int64).to_frame(),
            pl.Series("a", [2], dtype=pl.Int64).to_frame(),
            pl.Series("a", [], dtype=pl.Int64).to_frame(),
        ]
    )["a"].to_list() == [1, 2]


def test_sorted_flag_unset_by_arithmetic_4937() -> None:
    df = pl.DataFrame(
        {
            "ts": [1, 1, 1, 0, 1],
            "price": [3.3, 3.0, 3.5, 3.6, 3.7],
            "mask": [1, 1, 1, 1, 0],
        }
    )

    assert df.sort("price").group_by("ts").agg(
        [
            (pl.col("price") * pl.col("mask")).max().alias("pmax"),
            (pl.col("price") * pl.col("mask")).min().alias("pmin"),
        ]
    ).sort("ts").to_dict(as_series=False) == {
        "ts": [0, 1],
        "pmax": [3.6, 3.5],
        "pmin": [3.6, 0.0],
    }


def test_unset_sorted_flag_after_extend() -> None:
    df1 = pl.DataFrame({"Add": [37, 41], "Batch": [48, 49]}).sort("Add")
    df2 = pl.DataFrame({"Add": [37], "Batch": [67]}).sort("Add")

    df1.extend(df2)
    assert not df1["Add"].flags["SORTED_ASC"]
    df = df1.group_by("Add").agg([pl.col("Batch").min()]).sort("Add")
    assert df["Add"].flags["SORTED_ASC"]
    assert df.to_dict(as_series=False) == {"Add": [37, 41], "Batch": [48, 49]}


def test_sorted_flag_after_streaming_join() -> None:
    # streaming left join
    df1 = pl.DataFrame({"x": [1, 2, 3, 4], "y": [2, 4, 6, 6]}).set_sorted("x")
    df2 = pl.DataFrame({"x": [4, 2, 3, 1], "z": [1, 4, 9, 1]})
    assert (
        df1.lazy()
        .join(df2.lazy(), on="x", how="left")
        .collect(streaming=True)["x"]
        .flags["SORTED_ASC"]
    )


def test_sorted_flag_partition_by() -> None:
    assert (
        pl.DataFrame({"one": [1, 2, 3], "two": ["a", "a", "b"]})
        .set_sorted("one")
        .partition_by("two", maintain_order=True)[0]["one"]
        .flags["SORTED_ASC"]
    )


@pytest.mark.parametrize("value", [1, "a", True])
def test_sorted_flag_singletons(value: Any) -> None:
    assert pl.DataFrame({"x": [value]})["x"].flags["SORTED_ASC"] is True


def test_is_sorted() -> None:
    assert not pl.Series([1, 2, 5, None, 2, None]).is_sorted()
    assert pl.Series([1, 2, 4, None, None]).is_sorted(nulls_last=True)
    assert pl.Series([None, None, 1, 2, 4]).is_sorted(nulls_last=False)
    assert not pl.Series([None, 1, None, 2, 4]).is_sorted()
    assert not pl.Series([None, 1, 2, 3, -1, 4]).is_sorted(nulls_last=False)
    assert not pl.Series([1, 2, 3, -1, 4, None, None]).is_sorted(nulls_last=True)
    assert not pl.Series([1, 2, 3, -1, 4]).is_sorted()
    assert pl.Series([1, 2, 3, 4]).is_sorted()
    assert pl.Series([5, 2, 1, 1, -1]).is_sorted(descending=True)
    assert pl.Series([None, None, 5, 2, 1, 1, -1]).is_sorted(
        descending=True, nulls_last=False
    )
    assert pl.Series([5, 2, 1, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )
    assert not pl.Series([5, None, 2, 1, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )
    assert not pl.Series([5, 2, 1, 10, 1, -1, None, None]).is_sorted(
        descending=True, nulls_last=True
    )


def test_sorted_flag() -> None:
    s = pl.arange(0, 7, eager=True)
    assert s.flags["SORTED_ASC"]
    assert s.reverse().flags["SORTED_DESC"]
    assert pl.Series([b"a"]).set_sorted().flags["SORTED_ASC"]
    assert (
        pl.Series([date(2020, 1, 1), date(2020, 1, 2)])
        .set_sorted()
        .cast(pl.Datetime)
        .flags["SORTED_ASC"]
    )

    # empty
    q = pl.LazyFrame(
        schema={
            "store_id": pl.UInt16,
            "item_id": pl.UInt32,
            "timestamp": pl.Datetime,
        }
    ).sort("timestamp")

    assert q.collect()["timestamp"].flags["SORTED_ASC"]

    # ensure we don't panic for these types
    # struct
    pl.Series([{"a": 1}]).set_sorted(descending=True)
    # list
    pl.Series([[{"a": 1}]]).set_sorted(descending=True)
    # object
    pl.Series([{"a": 1}], dtype=pl.Object).set_sorted(descending=True)


def test_sorted_flag_after_joins() -> None:
    np.random.seed(1)
    dfa = pl.DataFrame(
        {
            "a": np.random.randint(0, 13, 20),
            "b": np.random.randint(0, 13, 20),
        }
    ).sort("a")

    dfb = pl.DataFrame(
        {
            "a": np.random.randint(0, 13, 10),
            "b": np.random.randint(0, 13, 10),
        }
    )

    dfapd = dfa.to_pandas()
    dfbpd = dfb.to_pandas()

    def test_with_pd(
        dfa: pd.DataFrame, dfb: pd.DataFrame, on: str, how: str, joined: pl.DataFrame
    ) -> None:
        a = (
            dfa.merge(
                dfb,
                on=on,
                how=how,  # type: ignore[arg-type]
                suffixes=("", "_right"),
            )
            .sort_values(["a", "b"])
            .reset_index(drop=True)
        )
        b = joined.sort(["a", "b"]).to_pandas()
        pd.testing.assert_frame_equal(a, b)

    joined = dfa.join(dfb, on="b", how="left", coalesce=True)
    assert joined["a"].flags["SORTED_ASC"]
    test_with_pd(dfapd, dfbpd, "b", "left", joined)

    joined = dfa.join(dfb, on="b", how="inner")
    assert joined["a"].flags["SORTED_ASC"]
    test_with_pd(dfapd, dfbpd, "b", "inner", joined)

    joined = dfa.join(dfb, on="b", how="semi")
    assert joined["a"].flags["SORTED_ASC"]
    joined = dfa.join(dfb, on="b", how="semi")
    assert joined["a"].flags["SORTED_ASC"]

    joined = dfb.join(dfa, on="b", how="left", coalesce=True)
    assert not joined["a"].flags["SORTED_ASC"]
    test_with_pd(dfbpd, dfapd, "b", "left", joined)

    joined = dfb.join(dfa, on="b", how="inner")
    assert not joined["a"].flags["SORTED_ASC"]

    joined = dfb.join(dfa, on="b", how="semi")
    assert not joined["a"].flags["SORTED_ASC"]
    joined = dfb.join(dfa, on="b", how="anti")
    assert not joined["a"].flags["SORTED_ASC"]


def test_sorted_flag_group_by_dynamic() -> None:
    df = pl.DataFrame({"ts": [date(2020, 1, 1), date(2020, 1, 2)], "val": [1, 2]})
    assert (
        (
            df.group_by_dynamic(pl.col("ts").set_sorted(), every="1d").agg(
                pl.col("val").sum()
            )
        )
        .to_series()
        .flags["SORTED_ASC"]
    )


def test_is_sorted_rle_id() -> None:
    assert pl.Series([12, 3345, 12, 3, 4, 4, 1, 12]).rle_id().flags["SORTED_ASC"]


def test_is_sorted_chunked_select() -> None:
    df = pl.DataFrame({"a": np.ones(14)})

    assert (
        pl.concat([df, df, df], rechunk=False)
        .set_sorted("a")
        .select(pl.col("a").alias("b"))
    )["b"].flags["SORTED_ASC"]
