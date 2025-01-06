from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import ColumnNotFoundError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_unique_predicate_pd() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["abc", "abc"],
            "y": ["xxx", "xxx"],
            "z": [True, False],
        }
    )

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="last")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame(schema={"x": pl.String, "y": pl.String, "z": pl.Boolean})
    assert_frame_equal(result, expected)

    result = (
        lf.unique(subset=["x", "y"], maintain_order=True, keep="any")
        .filter(pl.col("z"))
        .collect()
    )
    expected = pl.DataFrame({"x": ["abc"], "y": ["xxx"], "z": [True]})
    assert_frame_equal(result, expected)

    # Issue #14595: filter should not naively be pushed past unique()
    for maintain_order in (True, False):
        for keep in ("first", "last", "any", "none"):
            q = (
                lf.unique("x", maintain_order=maintain_order, keep=keep)
                .filter(pl.col("x") == "abc")
                .filter(pl.col("z"))
            )
            assert_frame_equal(q.collect(predicate_pushdown=False), q.collect())


def test_unique_on_list_df() -> None:
    assert pl.DataFrame(
        {"a": [1, 2, 3, 4, 4], "b": [[1, 1], [2], [3], [4, 4], [4, 4]]}
    ).unique(maintain_order=True).to_dict(as_series=False) == {
        "a": [1, 2, 3, 4],
        "b": [[1, 1], [2], [3], [4, 4]],
    }


def test_list_unique() -> None:
    s = pl.Series("a", [[1, 2], [3], [1, 2], [4, 5], [2], [2]])
    assert s.unique(maintain_order=True).to_list() == [[1, 2], [3], [4, 5], [2]]
    assert s.arg_unique().to_list() == [0, 1, 3, 4]
    assert s.n_unique() == 4


def test_unique_and_drop_stability() -> None:
    # see: 2898
    # the original cause was that we wrote:
    # expr_a = a.unique()
    # expr_a.filter(a.unique().is_not_null())
    # meaning that the a.unique was executed twice, which is an unstable algorithm
    df = pl.DataFrame({"a": [1, None, 1, None]})
    assert df.select(pl.col("a").unique().drop_nulls()).to_series()[0] == 1


def test_unique_empty() -> None:
    for dt in [pl.String, pl.Boolean, pl.Int32, pl.UInt32]:
        s = pl.Series([], dtype=dt)
        assert_series_equal(s.unique(), s)


def test_unique() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 2], "b": [3, 3, 3]})

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    assert_frame_equal(ldf.unique(maintain_order=True).collect(), expected)

    result = ldf.unique(subset="b", maintain_order=True).collect()
    expected = pl.DataFrame({"a": [1], "b": [3]})
    assert_frame_equal(result, expected)

    s0 = pl.Series("a", [1, 2, None, 2])
    # test if the null is included
    assert s0.unique().to_list() == [None, 1, 2]


def test_struct_unique_df() -> None:
    df = pl.DataFrame(
        {
            "numerical": [1, 2, 1],
            "struct": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}],
        }
    )

    df.select("numerical", "struct").unique().sort("numerical")


def test_sorted_unique_dates() -> None:
    assert (
        pl.DataFrame(
            [pl.Series("dt", [date(2015, 6, 24), date(2015, 6, 23)], dtype=pl.Date)]
        )
        .sort("dt")
        .unique()
    ).to_dict(as_series=False) == {"dt": [date(2015, 6, 23), date(2015, 6, 24)]}


def test_unique_null() -> None:
    s0 = pl.Series([])
    assert_series_equal(s0.unique(), s0)

    s1 = pl.Series([None])
    assert_series_equal(s1.unique(), s1)

    s2 = pl.Series([None, None])
    assert_series_equal(s2.unique(), s1)


@pytest.mark.parametrize(
    ("input", "output"),
    [
        ([], []),
        (["a", "b", "b", "c"], ["a", "b", "c"]),
        ([None, "a", "b", "b"], [None, "a", "b"]),
    ],
)
@pytest.mark.usefixtures("test_global_and_local")
def test_unique_categorical(input: list[str | None], output: list[str | None]) -> None:
    s = pl.Series(input, dtype=pl.Categorical)
    result = s.unique(maintain_order=True)
    expected = pl.Series(output, dtype=pl.Categorical)
    assert_series_equal(result, expected)

    result = s.unique(maintain_order=False).sort()
    expected = pl.Series(output, dtype=pl.Categorical)
    assert_series_equal(result, expected)


def test_unique_categorical_global() -> None:
    with pl.StringCache():
        pl.Series(["aaaa", "bbbb", "cccc"])  # pre-fill global cache
        s = pl.Series(["a", "b", "c"], dtype=pl.Categorical)
        s_empty = s.slice(0, 0)

        assert s_empty.unique().to_list() == []
        assert_series_equal(s_empty.cat.get_categories(), pl.Series(["a", "b", "c"]))


def test_unique_with_null() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 2, 3, 4],
            "b": ["a", "a", "b", "b", "c", "c"],
            "c": [None, None, None, None, None, None],
        }
    )
    expected_df = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["a", "b", "c", "c"], "c": [None, None, None, None]}
    )
    assert_frame_equal(df.unique(maintain_order=True), expected_df)


@pytest.mark.parametrize(
    ("input_json_data", "input_schema", "subset"),
    [
        ({"ID": [], "Name": []}, {"ID": pl.Int64, "Name": pl.String}, "id"),
        ({"ID": [], "Name": []}, {"ID": pl.Int64, "Name": pl.String}, ["age", "place"]),
        (
            {"ID": [1, 2, 1, 2], "Name": ["foo", "bar", "baz", "baa"]},
            {"ID": pl.Int64, "Name": pl.String},
            "id",
        ),
        (
            {"ID": [1, 2, 1, 2], "Name": ["foo", "bar", "baz", "baa"]},
            {"ID": pl.Int64, "Name": pl.String},
            ["age", "place"],
        ),
    ],
)
def test_unique_with_bad_subset(
    input_json_data: dict[str, list[Any]],
    input_schema: dict[str, PolarsDataType],
    subset: str | list[str],
) -> None:
    df = pl.DataFrame(input_json_data, schema=input_schema)

    with pytest.raises(ColumnNotFoundError, match="not found"):
        df.unique(subset=subset)


@pytest.mark.usefixtures("test_global_and_local")
def test_categorical_unique_19409() -> None:
    df = pl.DataFrame({"x": [str(n % 50) for n in range(127)]}).cast(pl.Categorical)
    uniq = df.unique()
    assert uniq.height == 50
    assert uniq.null_count().item() == 0
    assert set(uniq["x"]) == set(df["x"])


def test_categorical_updated_revmap_unique_20233() -> None:
    with pl.StringCache():
        s = pl.Series("a", ["A"], pl.Categorical)

        s = (
            pl.select(a=pl.when(True).then(pl.lit("C", pl.Categorical)))
            .select(a=pl.when(True).then(pl.lit("D", pl.Categorical)))
            .to_series()
        )

        assert_series_equal(s.unique(), pl.Series("a", ["D"], pl.Categorical))


def test_unique_check_order_20480() -> None:
    df = pl.DataFrame(
        [
            {
                "key": "some_key",
                "value": "second",
                "number": 2,
            },
            {
                "key": "some_key",
                "value": "first",
                "number": 1,
            },
        ]
    )
    assert (
        df.lazy()
        .sort("key", "number")
        .unique(subset="key", keep="first")
        .collect()["number"]
        .item()
        == 1
    )


def test_predicate_pushdown_unique() -> None:
    q = (
        pl.LazyFrame({"id": [1, 2, 3]})
        .with_columns(pl.date(2024, 1, 1) + pl.duration(days=[1, 2, 3]))  # type: ignore[arg-type]
        .unique()
    )

    print(q.filter(pl.col("id").is_in([1, 2, 3])).explain())
    assert not q.filter(pl.col("id").is_in([1, 2, 3])).explain().startswith("FILTER")
    assert q.filter(pl.col("id").sum() == pl.col("id")).explain().startswith("FILTER")
