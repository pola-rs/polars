from datetime import datetime

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_frame_equal


def test_with_context() -> None:
    df_a = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "c", None]}).lazy()
    df_b = pl.DataFrame({"c": ["foo", "ham"]})

    with pytest.deprecated_call():
        result = df_a.with_context(df_b.lazy()).select(
            pl.col("b") + pl.col("c").first()
        )
    assert result.collect().to_dict(as_series=False) == {"b": ["afoo", "cfoo", None]}

    with pytest.deprecated_call():
        context = df_a.with_context(df_b.lazy())
    with pytest.raises(ComputeError):
        context.select("a", "c").collect()


# https://github.com/pola-rs/polars/issues/5867
def test_with_context_ignore_5867() -> None:
    outer = pl.LazyFrame({"OtherCol": [1, 2, 3, 4]})
    with pytest.deprecated_call():
        lf = pl.LazyFrame(
            {"Category": [1, 1, 2, 2], "Counts": [1, 2, 3, 4]}
        ).with_context(outer)

    result = lf.group_by("Category", maintain_order=True).agg(pl.col("Counts").sum())

    expected = pl.LazyFrame({"Category": [1, 2], "Counts": [3, 7]})
    assert_frame_equal(result, expected)


def test_predicate_pushdown_with_context_11014() -> None:
    df1 = pl.LazyFrame(
        {
            "df1_c1": [1, 2, 3],
            "df1_c2": [2, 3, 4],
        }
    )

    df2 = pl.LazyFrame(
        {
            "df2_c1": [2, 3, 4],
            "df2_c2": [3, 4, 5],
        }
    )

    with pytest.deprecated_call():
        out = (
            df1.with_context(df2)
            .filter(pl.col("df1_c1").is_in(pl.col("df2_c1")))
            .collect(predicate_pushdown=True)
        )

    assert out.to_dict(as_series=False) == {"df1_c1": [2, 3], "df1_c2": [3, 4]}


@pytest.mark.xdist_group("streaming")
def test_streaming_11219() -> None:
    # https://github.com/pola-rs/polars/issues/11219

    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["a", "c", None]})
    lf_other = pl.LazyFrame({"c": ["foo", "ham"]})
    lf_other2 = pl.LazyFrame({"c": ["foo", "ham"]})

    with pytest.deprecated_call():
        context = lf.with_context([lf_other, lf_other2])

    assert context.select(pl.col("b") + pl.col("c").first()).collect(
        streaming=True
    ).to_dict(as_series=False) == {"b": ["afoo", "cfoo", None]}


def test_no_cse_in_with_context() -> None:
    df1 = pl.DataFrame(
        {
            "timestamp": [
                datetime(2023, 1, 1, 0, 0),
                datetime(2023, 5, 1, 0, 0),
                datetime(2023, 10, 1, 0, 0),
            ],
            "value": [2, 5, 9],
        }
    )
    df2 = pl.DataFrame(
        {
            "date_start": [
                datetime(2022, 12, 31, 0, 0),
                datetime(2023, 1, 2, 0, 0),
            ],
            "date_end": [
                datetime(2023, 4, 30, 0, 0),
                datetime(2023, 5, 5, 0, 0),
            ],
            "label": [0, 1],
        }
    )

    with pytest.deprecated_call():
        context = df1.lazy().with_context(df2.lazy())

    assert (
        context.select(
            pl.col("date_start", "label").gather(
                pl.col("date_start").search_sorted(pl.col("timestamp")) - 1
            ),
        )
    ).collect().to_dict(as_series=False) == {
        "date_start": [
            datetime(2022, 12, 31, 0, 0),
            datetime(2023, 1, 2, 0, 0),
            datetime(2023, 1, 2, 0, 0),
        ],
        "label": [0, 1, 1],
    }
