import pytest

import polars as pl
import polars.selectors as cs
from polars import StringCache
from polars.testing import assert_frame_equal


def test_unpivot() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    expected = {
        ("a", "B", 1),
        ("b", "B", 3),
        ("c", "B", 5),
        ("a", "C", 2),
        ("b", "C", 4),
        ("c", "C", 6),
    }
    for _idv, _vv in (("A", ("B", "C")), (cs.string(), cs.integer())):
        unpivoted_eager = df.unpivot(index="A", on=["B", "C"])
        assert set(unpivoted_eager.iter_rows()) == expected

        unpivoted_lazy = df.lazy().unpivot(index="A", on=["B", "C"]).collect()
        assert set(unpivoted_lazy.iter_rows()) == expected

    unpivoted = df.unpivot(index="A", on="B")
    assert set(unpivoted["value"]) == {1, 3, 5}

    expected_full = {
        ("A", "a"),
        ("A", "b"),
        ("A", "c"),
        ("B", "1"),
        ("B", "3"),
        ("B", "5"),
        ("C", "2"),
        ("C", "4"),
        ("C", "6"),
    }
    for unpivoted in [df.unpivot(), df.lazy().unpivot().collect()]:
        assert set(unpivoted.iter_rows()) == expected_full

    with pytest.deprecated_call(match="unpivot"):
        for unpivoted in [
            df.melt(value_name="foo", variable_name="bar"),
            df.lazy().melt(value_name="foo", variable_name="bar").collect(),
        ]:
            assert set(unpivoted.iter_rows()) == expected_full


def test_unpivot_projection_pd_7747() -> None:
    df = pl.LazyFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "age": [40, 30, 21, 33, 45],
            "weight": [100, 103, 95, 90, 110],
        }
    )
    with pytest.deprecated_call(match="unpivot"):
        result = (
            df.with_columns(pl.col("age").alias("wgt"))
            .melt(id_vars="number", value_vars="wgt")
            .select("number", "value")
            .collect()
        )
    expected = pl.DataFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "value": [40, 30, 21, 33, 45],
        }
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/10075
def test_unpivot_no_on() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    result = lf.unpivot(index="a")

    expected = pl.LazyFrame(
        schema={"a": pl.Int64, "variable": pl.String, "value": pl.Null}
    )
    assert_frame_equal(result, expected)


def test_unpivot_raise_list() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.LazyFrame(
            {"a": ["x", "y"], "b": [["test", "test2"], ["test3", "test4"]]}
        ).unpivot().collect()


def test_unpivot_empty_18170() -> None:
    assert pl.DataFrame().unpivot().schema == pl.Schema(
        {"variable": pl.String(), "value": pl.Null()}
    )


@StringCache()
def test_unpivot_categorical_global() -> None:
    df = pl.DataFrame(
        {
            "index": [0, 1],
            "1": pl.Series(["a", "b"], dtype=pl.Categorical),
            "2": pl.Series(["b", "c"], dtype=pl.Categorical),
        }
    )
    out = df.unpivot(["1", "2"], index="index")
    assert out.dtypes == [pl.Int64, pl.String, pl.Categorical(ordering="physical")]
    assert out.to_dict(as_series=False) == {
        "index": [0, 1, 0, 1],
        "variable": ["1", "1", "2", "2"],
        "value": ["a", "b", "b", "c"],
    }


@pytest.mark.may_fail_auto_streaming
def test_unpivot_categorical_raise_19770() -> None:
    with pytest.raises(pl.exceptions.ComputeError):
        (pl.DataFrame({"x": ["foo"]}).cast(pl.Categorical).unpivot())
