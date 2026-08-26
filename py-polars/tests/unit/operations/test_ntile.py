from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_ntile_no_partition() -> None:
    df = pl.DataFrame({"v": [10, 20, 30, 40, 50]})
    res = df.select(pl.ntile(2).alias("nt"))
    assert res["nt"].to_list() == [1, 1, 1, 2, 2]
    assert res["nt"].dtype == pl.get_index_type()
    assert res["nt"].null_count() == 0


def test_ntile_over_partition_and_order() -> None:
    df = pl.DataFrame({"lbl": ["A", "A", "A", "B", "B"], "value": [3, 1, 2, 9, 4]})
    res = df.with_columns(nt=pl.ntile(2).over("lbl", order_by="value"))
    assert res["nt"].to_list() == [2, 1, 1, 2, 1]


@pytest.mark.parametrize("n", [2, 3, 5, 100])
def test_ntile_at_least_as_many_buckets_as_rows(n: int) -> None:
    # n >= r is legal: each row becomes its own bucket, trailing buckets are empty
    df = pl.DataFrame({"v": [10, 20]})
    assert df.select(pl.ntile(n))["ntile"].to_list() == [1, 2]


def test_ntile_bucket_size_matrix() -> None:
    # check/validate `r < n`, `r > n`, and `r == n`
    for r in range(1, 25):
        df = pl.DataFrame({"v": range(r)})
        for n in range(1, 25):
            actual = df.select(pl.ntile(n).over(order_by="v"))["ntile"].to_list()
            quotient, remainder = divmod(r, n)
            expected = [
                bucket
                for bucket in range(1, n + 1)
                for _ in range(quotient + (bucket <= remainder))
            ]
            assert actual == expected, f"mismatch for r={r}, n={n}"


def test_ntile_single_bucket() -> None:
    df = pl.DataFrame({"v": [1, 2, 3]})
    assert df.select(pl.ntile(1).alias("nt"))["nt"].to_list() == [1, 1, 1]


def test_ntile_empty_frame() -> None:
    df = pl.DataFrame({"v": []}, schema={"v": pl.Int64})
    res = df.select(pl.ntile(4).alias("nt"))
    assert res.height == 0
    assert res["nt"].dtype == pl.get_index_type()


def test_ntile_ignores_value_ties() -> None:
    # ntile is positional: tied values may land in different buckets
    df = pl.DataFrame({"v": [5, 5, 5, 5]})
    assert df.select(pl.ntile(2).over(order_by="v"))["ntile"].to_list() == [1, 1, 2, 2]


def test_ntile_nulls_in_order_key_still_bucketed() -> None:
    # every row gets a bucket; nulls only affect *where* they sort
    df = pl.DataFrame({"v": [None, 5, None, 1]})
    res = df.with_columns(nt=pl.ntile(2).over(order_by="v", nulls_last=True))
    assert res["nt"].null_count() == 0
    assert sorted(res["nt"].to_list()) == [1, 1, 2, 2]


def test_ntile_matches_across_engines() -> None:
    df = pl.DataFrame({"lbl": ["A", "A", "A", "B", "B"], "value": [3, 1, 2, 9, 4]})

    ntile_expr = pl.ntile(2).over("lbl", order_by="value")
    res_in_mem = df.with_columns(nt=ntile_expr)
    res_streaming = df.lazy().with_columns(nt=ntile_expr).collect()

    assert_frame_equal(res_in_mem, res_streaming)
    for res in (res_in_mem, res_streaming):
        assert res["nt"].to_list() == [2, 1, 1, 2, 1]


@pytest.mark.parametrize("n", [0, -1, -100])
def test_ntile_invalid_n_raises(n: int) -> None:
    with pytest.raises(
        (pl.exceptions.InvalidOperationError, OverflowError, ValueError)
    ):
        pl.ntile(n)
