import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_nan_in_group_by_agg() -> None:
    df = pl.DataFrame(
        {
            "key": ["a", "a", "a", "a"],
            "value": [18.58, 18.78, float("nan"), 18.63],
            "bar": [0, 0, 0, 0],
        }
    )

    assert df.group_by("bar", "key").agg(pl.col("value").max())["value"].item() == 18.78
    assert df.group_by("bar", "key").agg(pl.col("value").min())["value"].item() == 18.58


def test_nan_aggregations() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 2.0, 3.0], "b": [1, 1, 1, 1]})

    aggs = [
        pl.col("a").max().alias("max"),
        pl.col("a").min().alias("min"),
        pl.col("a").nan_max().alias("nan_max"),
        pl.col("a").nan_min().alias("nan_min"),
    ]

    assert (
        str(df.select(aggs).to_dict(as_series=False))
        == "{'max': [3.0], 'min': [1.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )
    assert (
        str(df.group_by("b").agg(aggs).to_dict(as_series=False))
        == "{'b': [1], 'max': [3.0], 'min': [1.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )


@pytest.mark.parametrize("descending", [True, False])
def test_sorted_nan_max_12931(descending: bool) -> None:
    s = pl.Series("x", [1.0, 2.0, float("nan")]).sort(descending=descending)

    assert s.max() == 2.0
    assert s.arg_max() == 1

    # Test full-nan
    s = pl.Series("x", [float("nan"), float("nan"), float("nan")]).sort(
        descending=descending
    )

    out = s.max()
    assert out != out

    # This is flipped because float arg_max calculates the index as
    # * sorted ascending: (index of left-most NaN) - 1, saturating subtraction at 0
    # * sorted descending: (index of right-most NaN) + 1, saturating addition at s.len()
    assert s.arg_max() == (0, 2)[descending]

    s = pl.Series("x", [1.0, 2.0, 3.0]).sort(descending=descending)

    assert s.max() == 3.0
    assert s.arg_max() == (2, 0)[descending]


@pytest.mark.parametrize(
    ("s", "expect"),
    [
        (
            pl.Series(
                "x",
                [
                    -0.0,
                    0.0,
                    float("-nan"),
                    float("nan"),
                    1.0,
                    None,
                ],
            ),
            pl.Series("x", [None, 0.0, 1.0, float("nan")]),
        ),
        (
            # No nulls
            pl.Series(
                "x",
                [
                    -0.0,
                    0.0,
                    float("-nan"),
                    float("nan"),
                    1.0,
                ],
            ),
            pl.Series("x", [0.0, 1.0, float("nan")]),
        ),
    ],
)
def test_unique(s: pl.Series, expect: pl.Series) -> None:
    out = s.unique()
    assert_series_equal(expect, out)

    out = s.n_unique()  # type: ignore[assignment]
    assert expect.len() == out

    out = s.gather(s.arg_unique()).sort()
    assert_series_equal(expect, out)


def test_unique_counts() -> None:
    s = pl.Series(
        "x",
        [
            -0.0,
            0.0,
            float("-nan"),
            float("nan"),
            1.0,
            None,
        ],
    )
    expect = pl.Series("x", [2, 2, 1, 1], dtype=pl.UInt32)
    out = s.unique_counts()
    assert_series_equal(expect, out)


def test_hash() -> None:
    s = pl.Series(
        "x",
        [
            -0.0,
            0.0,
            float("-nan"),
            float("nan"),
            1.0,
            None,
        ],
    ).hash()

    # check them against each other since hash is not stable
    assert s.item(0) == s.item(1)  # hash(-0.0) == hash(0.0)
    assert s.item(2) == s.item(3)  # hash(float('-nan')) == hash(float('nan'))


def test_group_by_float() -> None:
    # Test num_groups_proxy
    # * -0.0 and 0.0 in same groups
    # * -nan and nan in same groups
    df = (
        pl.Series(
            "x",
            [
                -0.0,
                0.0,
                float("-nan"),
                float("nan"),
                1.0,
                None,
            ],
        )
        .to_frame()
        .with_row_index()
        .with_columns(a=pl.lit("a"))
    )

    expect = pl.Series("index", [[0, 1], [2, 3], [4], [5]], dtype=pl.List(pl.UInt32))
    expect_no_null = expect.head(3)

    for group_keys in (("x",), ("x", "a")):
        for maintain_order in (True, False):
            for drop_nulls in (True, False):
                out = df
                if drop_nulls:
                    out = out.drop_nulls()

                out = (
                    out.group_by(group_keys, maintain_order=maintain_order)  # type: ignore[assignment]
                    .agg("index")
                    .sort(pl.col("index").list.get(0))
                    .select("index")
                    .to_series()
                )

                if drop_nulls:
                    assert_series_equal(expect_no_null, out)  # type: ignore[arg-type]
                else:
                    assert_series_equal(expect, out)  # type: ignore[arg-type]


def test_joins() -> None:
    # Test that -0.0 joins with 0.0 and nan joins with nan
    df = (
        pl.Series(
            "x",
            [
                -0.0,
                0.0,
                float("-nan"),
                float("nan"),
                1.0,
                None,
            ],
        )
        .to_frame()
        .with_row_index()
        .with_columns(a=pl.lit("a"))
    )

    rhs = (
        pl.Series("x", [0.0, float("nan"), 3.0])
        .to_frame()
        .with_columns(a=pl.lit("a"), rhs=True)
    )

    for join_on in (
        # Single and multiple keys
        ("x",),
        (
            "x",
            "a",
        ),
    ):
        how = "left"
        expect = pl.Series("rhs", [True, True, True, True, None, None])
        out = df.join(rhs, on=join_on, how=how).sort("index").select("rhs").to_series()  # type: ignore[arg-type]
        assert_series_equal(expect, out)

        how = "inner"
        expect = pl.Series("index", [0, 1, 2, 3], dtype=pl.UInt32)
        out = (
            df.join(rhs, on=join_on, how=how).sort("index").select("index").to_series()  # type: ignore[arg-type]
        )
        assert_series_equal(expect, out)

        how = "full"
        expect = pl.Series("rhs", [True, True, True, True, None, None, True])
        out = (
            df.join(rhs, on=join_on, how=how)  # type: ignore[arg-type]
            .sort("index", nulls_last=True)
            .select("rhs")
            .to_series()
        )
        assert_series_equal(expect, out)

        how = "semi"
        expect = pl.Series("x", [-0.0, 0.0, float("-nan"), float("nan")])
        out = (
            df.join(rhs, on=join_on, how=how)  # type: ignore[arg-type]
            .sort("index", nulls_last=True)
            .select("x")
            .to_series()
        )
        assert_series_equal(expect, out)

        how = "anti"
        expect = pl.Series("x", [1.0, None])
        out = (
            df.join(rhs, on=join_on, how=how)  # type: ignore[arg-type]
            .sort("index", nulls_last=True)
            .select("x")
            .to_series()
        )
        assert_series_equal(expect, out)

    # test asof
    # note that nans never join because nans are always greater than the other
    # side of the comparison (i.e. NaN > tolerance)
    expect = pl.Series("rhs", [True, True, None, None, None, None])
    out = (
        df.sort("x")
        .join_asof(rhs.sort("x"), on="x", tolerance=0)
        .sort("index")
        .select("rhs")
        .to_series()
    )
    assert_series_equal(expect, out)


def test_first_last_distinct() -> None:
    s = pl.Series(
        "x",
        [
            -0.0,
            0.0,
            float("-nan"),
            float("nan"),
            1.0,
            None,
        ],
    )

    assert_series_equal(
        pl.Series("x", [True, False, True, False, True, True]), s.is_first_distinct()
    )

    assert_series_equal(
        pl.Series("x", [False, True, False, True, True, True]), s.is_last_distinct()
    )
