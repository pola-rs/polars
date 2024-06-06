from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_list_set_oob() -> None:
    df = pl.DataFrame({"a": [[42], [23]]})
    result = df.select(pl.col("a").list.set_intersection([]))
    assert result.to_dict(as_series=False) == {"a": [[], []]}


def test_list_set_operations_float() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2, 3], [1, 1, 1], [4]], "b": [[4, 2, 1], [2, 1, 12], [4]]},
        schema={"a": pl.List(pl.Float32), "b": pl.List(pl.Float32)},
    )

    assert df.select(pl.col("a").list.set_union("b"))["a"].to_list() == [
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 12.0],
        [4.0],
    ]
    assert df.select(pl.col("a").list.set_intersection("b"))["a"].to_list() == [
        [1.0, 2.0],
        [1.0],
        [4.0],
    ]
    assert df.select(pl.col("a").list.set_difference("b"))["a"].to_list() == [
        [3.0],
        [],
        [],
    ]
    assert df.select(pl.col("b").list.set_difference("a"))["b"].to_list() == [
        [4.0],
        [2.0, 12.0],
        [],
    ]


def test_list_set_operations() -> None:
    df = pl.DataFrame(
        {"a": [[1, 2, 3], [1, 1, 1], [4]], "b": [[4, 2, 1], [2, 1, 12], [4]]}
    )

    assert df.select(pl.col("a").list.set_union("b"))["a"].to_list() == [
        [1, 2, 3, 4],
        [1, 2, 12],
        [4],
    ]
    assert df.select(pl.col("a").list.set_intersection("b"))["a"].to_list() == [
        [1, 2],
        [1],
        [4],
    ]
    assert df.select(pl.col("a").list.set_difference("b"))["a"].to_list() == [
        [3],
        [],
        [],
    ]
    assert df.select(pl.col("b").list.set_difference("a"))["b"].to_list() == [
        [4],
        [2, 12],
        [],
    ]

    # check logical types
    dtype = pl.List(pl.Date)
    assert (
        df.select(pl.col("b").cast(dtype).list.set_difference(pl.col("a").cast(dtype)))[
            "b"
        ].dtype
        == dtype
    )

    df = pl.DataFrame(
        {
            "a": [["a", "b", "c"], ["b", "e", "z"]],
            "b": [["b", "s", "a"], ["a", "e", "f"]],
        }
    )

    assert df.select(pl.col("a").list.set_union("b"))["a"].to_list() == [
        ["a", "b", "c", "s"],
        ["b", "e", "z", "a", "f"],
    ]

    df = pl.DataFrame(
        {
            "a": [[2, 3, 3], [3, 1], [1, 2, 3]],
            "b": [[2, 3, 4], [3, 3, 1], [3, 3]],
        }
    )
    r1 = df.with_columns(pl.col("a").list.set_intersection("b"))["a"].to_list()
    r2 = df.with_columns(pl.col("b").list.set_intersection("a"))["b"].to_list()
    exp = [[2, 3], [3, 1], [3]]
    assert r1 == exp
    assert r2 == exp


def test_list_set_operations_broadcast() -> None:
    df = pl.DataFrame(
        {
            "a": [[2, 3, 3], [3, 1], [1, 2, 3]],
        }
    )

    assert df.with_columns(
        pl.col("a").list.set_intersection(pl.lit(pl.Series([[1, 2]])))
    ).to_dict(as_series=False) == {"a": [[2], [1], [1, 2]]}
    assert df.with_columns(
        pl.col("a").list.set_union(pl.lit(pl.Series([[1, 2]])))
    ).to_dict(as_series=False) == {"a": [[2, 3, 1], [3, 1, 2], [1, 2, 3]]}
    assert df.with_columns(
        pl.col("a").list.set_difference(pl.lit(pl.Series([[1, 2]])))
    ).to_dict(as_series=False) == {"a": [[3], [3], [3]]}
    assert df.with_columns(
        pl.lit(pl.Series("a", [[1, 2]])).list.set_difference("a")
    ).to_dict(as_series=False) == {"a": [[1], [2], []]}


def test_list_set_operation_different_length_chunk_12734() -> None:
    df = pl.DataFrame(
        {
            "a": [[2, 3, 3], [4, 1], [1, 2, 3]],
        }
    )

    df = pl.concat([df.slice(0, 1), df.slice(1, 1), df.slice(2, 1)], rechunk=False)
    assert df.with_columns(
        pl.col("a").list.set_difference(pl.lit(pl.Series([[1, 2]])))
    ).to_dict(as_series=False) == {"a": [[3], [4], [3]]}


def test_list_set_operations_binary() -> None:
    df = pl.DataFrame(
        {
            "a": [[b"1", b"2", b"3"], [b"1", b"1", b"1"], [b"4"]],
            "b": [[b"4", b"2", b"1"], [b"2", b"1", b"12"], [b"4"]],
        },
        schema={"a": pl.List(pl.Binary), "b": pl.List(pl.Binary)},
    )

    assert df.select(pl.col("a").list.set_union("b"))["a"].to_list() == [
        [b"1", b"2", b"3", b"4"],
        [b"1", b"2", b"12"],
        [b"4"],
    ]
    assert df.select(pl.col("a").list.set_intersection("b"))["a"].to_list() == [
        [b"1", b"2"],
        [b"1"],
        [b"4"],
    ]
    assert df.select(pl.col("a").list.set_difference("b"))["a"].to_list() == [
        [b"3"],
        [],
        [],
    ]
    assert df.select(pl.col("b").list.set_difference("a"))["b"].to_list() == [
        [b"4"],
        [b"2", b"12"],
        [],
    ]


def test_set_operations_14290() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, 2], [2, 3]],
            "b": [None, [1, 2]],
        }
    )

    out = df.with_columns(pl.col("a").shift(1).alias("shifted_a")).select(
        b_dif_a=pl.col("b").list.set_difference("a"),
        shifted_a_dif_a=pl.col("shifted_a").list.set_difference("a"),
    )
    expected = pl.DataFrame({"b_dif_a": [None, [1]], "shifted_a_dif_a": [None, [1]]})
    assert_frame_equal(out, expected)


def test_broadcast_sliced() -> None:
    df = pl.DataFrame({"a": [[1, 2], [3, 4]]})
    out = df.select(
        pl.col("a").list.set_difference(pl.Series([[1], [2, 3, 4]]).slice(0, 1))
    )
    expected = pl.DataFrame({"a": [[2], [3, 4]]})

    assert_frame_equal(out, expected)
