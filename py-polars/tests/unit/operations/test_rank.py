import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_rank_nulls() -> None:
    assert pl.Series([]).rank().to_list() == []
    assert pl.Series([None]).rank().to_list() == [None]
    assert pl.Series([None, None]).rank().to_list() == [None, None]


def test_rank_random_expr() -> None:
    df = pl.from_dict(
        {"a": [1] * 5, "b": [1, 2, 3, 4, 5], "c": [200, 100, 100, 50, 100]}
    )

    df_ranks1 = df.with_columns(
        pl.col("c").rank(method="random", seed=1).over("a").alias("rank")
    )
    df_ranks2 = df.with_columns(
        pl.col("c").rank(method="random", seed=1).over("a").alias("rank")
    )
    assert_frame_equal(df_ranks1, df_ranks2)


def test_rank_random_series() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])
    assert_series_equal(
        s.rank("random", seed=1),
        pl.Series("a", [2, 5, 7, 3, 4, 6, 1], dtype=pl.get_index_type()),
    )


def test_rank_df() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
        }
    )

    s = df.select(pl.col("a").rank(method="average").alias("b")).to_series()
    assert s.to_list() == [1.5, 1.5, 3.5, 3.5, 5.0]
    assert s.dtype == pl.Float64

    s = df.select(pl.col("a").rank(method="max").alias("b")).to_series()
    assert s.to_list() == [2, 2, 4, 4, 5]
    assert s.dtype == pl.get_index_type()


@pytest.mark.parametrize("maintain_order", [False, True])
def test_rank_so_4109(maintain_order: bool) -> None:
    # also tests ranks null behavior
    df = pl.from_dict(
        {
            "id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "rank": [None, 3, 2, 4, 1, 4, 3, 2, 1, None, 3, 4, 4, 1, None, 3],
        }
    ).sort(by=["id", "rank"])

    df = df.group_by("id", maintain_order=maintain_order).agg(
        [
            pl.col("rank").alias("original"),
            pl.col("rank").rank(method="dense").alias("dense"),
            pl.col("rank").rank(method="average").alias("average"),
        ]
    )
    expected = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "original": [
                [None, 2, 3, 4],
                [1, 2, 3, 4],
                [None, 1, 3, 4],
                [None, 1, 3, 4],
            ],
            "dense": [
                [None, 1, 2, 3],
                [1, 2, 3, 4],
                [None, 1, 2, 3],
                [None, 1, 2, 3],
            ],
            "average": [
                [None, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [None, 1.0, 2.0, 3.0],
                [None, 1.0, 2.0, 3.0],
            ],
        },
        schema=df.schema,
    )

    assert_frame_equal(df, expected, check_row_order=maintain_order)


def test_rank_string_null_11252() -> None:
    rank = pl.Series([None, "", "z", None, "a"]).rank()
    assert rank.to_list() == [None, 1.0, 3.0, None, 2.0]


def test_rank_series() -> None:
    s = pl.Series("a", [1, 2, 3, 2, 2, 3, 0])

    assert_series_equal(
        s.rank("dense"),
        pl.Series("a", [2, 3, 4, 3, 3, 4, 1], dtype=pl.get_index_type()),
    )

    df = pl.DataFrame([s])
    assert df.select(pl.col("a").rank("dense"))["a"].to_list() == [2, 3, 4, 3, 3, 4, 1]

    assert_series_equal(
        s.rank("dense", descending=True),
        pl.Series("a", [3, 2, 1, 2, 2, 1, 4], dtype=pl.get_index_type()),
    )

    assert s.rank(method="average").dtype == pl.Float64
    assert s.rank(method="max").dtype == pl.get_index_type()


@pytest.mark.parametrize(
    ("value", "dtype"),
    [
        (5, pl.Int64),
        (2.5, pl.Float64),
        (True, pl.Boolean),
        ("abcdef", pl.String),
    ],
)
@pytest.mark.parametrize(
    "method", ["average", "min", "max", "dense", "ordinal", "random"]
)
@pytest.mark.parametrize("descending", [False, True])
def test_rank_values_that_repeat_under_a_mask(
    value: object, dtype: pl.DataType, method: str, descending: bool
) -> None:
    # Every element the mask says is there holds the same value, so they are one tie
    # group and the answer is one rank under the column's own mask. The column does not
    # repeat one *element* -- the mask makes some of them null -- so the scalar answer
    # has to be read off the values axis alone.
    n = 999
    masked = pl.select(
        pl.when(pl.int_range(0, n) % 3 != 0)
        .then(pl.repeat(pl.lit(value, dtype=dtype), n))
        .alias("a")
    ).to_series()
    written = pl.Series("a", masked.to_list(), dtype=dtype)

    ranked = masked.rank(method, descending=descending, seed=1)  # type: ignore[arg-type]
    assert_series_equal(
        ranked,
        written.rank(method, descending=descending, seed=1),  # type: ignore[arg-type]
    )

    # A rank of one repeated value is one rank: it is held once, not once per element.
    if method in ("average", "min", "max", "dense"):
        assert ranked.estimated_size() < written.estimated_size()
