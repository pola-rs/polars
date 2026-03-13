import datetime
from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
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


def test_unpivot_categorical() -> None:
    df = pl.DataFrame(
        {
            "index": [0, 1],
            "1": pl.Series(["a", "b"], dtype=pl.Categorical),
            "2": pl.Series(["b", "c"], dtype=pl.Categorical),
        }
    )
    out = df.unpivot(["1", "2"], index="index")
    assert out.dtypes == [pl.Int64, pl.String, pl.Categorical()]
    assert out.to_dict(as_series=False) == {
        "index": [0, 1, 0, 1],
        "variable": ["1", "1", "2", "2"],
        "value": ["a", "b", "b", "c"],
    }


def test_unpivot_index_not_found_23165() -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        pl.DataFrame({"a": [1]}).unpivot(index="b")


def assert_eq_df_lf_impl(
    data: Any, expr: Any, on: Any, index: Any, expected_data: list[pl.Series]
) -> None:
    df_result = expr(pl.DataFrame(data), on, index)
    lf_result = expr(pl.LazyFrame(data), on, index).collect()
    expected_result = pl.DataFrame(expected_data)

    assert_frame_equal(df_result, lf_result, check_row_order=False)
    assert_frame_equal(df_result, expected_result, check_row_order=False)


def test_unpivot_empty_on_25474() -> None:
    data = {
        "a": ["x", "y"],
        "b": [1, 3],
        "c": [2, 4],
        "d": ["str_a", "str_b"],
    }

    def assert_eq_df_lf(on: Any, index: Any, expected_data: list[pl.Series]) -> None:
        def logic(frame: Any, on: Any, index: Any) -> Any:
            return frame.unpivot(on, index=index, variable_name="var", value_name="val")

        return assert_eq_df_lf_impl(data, logic, on, index, expected_data)

    assert_eq_df_lf(
        pl.selectors.numeric(),
        "a",
        [
            pl.Series("a", ["x", "y", "x", "y"], dtype=pl.String),
            pl.Series("var", ["b", "b", "c", "c"], dtype=pl.String),
            pl.Series("val", [1, 3, 2, 4], dtype=pl.Int64),
        ],
    )

    assert_eq_df_lf(
        pl.selectors.date(),
        "a",
        [
            pl.Series("a", [], dtype=pl.String),
            pl.Series("var", [], dtype=pl.String),
            pl.Series("val", [], dtype=pl.Null),
        ],
    )

    assert_eq_df_lf(
        pl.selectors.date(),
        "b",
        [
            pl.Series("b", [], dtype=pl.Int64),
            pl.Series("var", [], dtype=pl.String),
            pl.Series("val", [], dtype=pl.Null),
        ],
    )

    assert_eq_df_lf(
        [],
        "a",
        [
            pl.Series("a", [], dtype=pl.String),
            pl.Series("var", [], dtype=pl.String),
            pl.Series("val", [], dtype=pl.Null),
        ],
    )

    assert_eq_df_lf(
        None,
        "a",
        [
            pl.Series("a", ["x", "y", "x", "y", "x", "y"], dtype=pl.String),
            pl.Series("var", ["b", "b", "c", "c", "d", "d"], dtype=pl.String),
            pl.Series("val", ["1", "3", "2", "4", "str_a", "str_b"], dtype=pl.String),
        ],
    )

    assert_eq_df_lf(
        None,
        ["b", "a"],
        [
            pl.Series("b", [1, 3, 1, 3], dtype=pl.Int64),
            pl.Series("a", ["x", "y", "x", "y"], dtype=pl.String),
            pl.Series("var", ["c", "c", "d", "d"], dtype=pl.String),
            pl.Series("val", ["2", "4", "str_a", "str_b"], dtype=pl.String),
        ],
    )


def test_unpivot_predicate_pd() -> None:
    day_a = datetime.date(2995, 4, 3)
    day_b = datetime.date(2333, 4, 3)

    data = {
        "a": ["x", "y", "z"],
        "b": [1, 3, 1],
        "c": [2, 4, 7],
        "d": [day_a, day_a, day_b],
    }

    def assert_eq_df_lf(on: Any, index: Any, expected_data: list[pl.Series]) -> None:
        def logic(frame: Any, on: Any, index: Any) -> Any:
            return frame.unpivot(on, index=index).filter(pl.col.b == 1)

        return assert_eq_df_lf_impl(data, logic, on, index, expected_data)

    assert_eq_df_lf(
        None,
        ["b", "a"],
        [
            pl.Series("b", [1, 1, 1, 1], dtype=pl.Int64),
            pl.Series("a", ["x", "z", "x", "z"], dtype=pl.String),
            pl.Series("variable", ["c", "c", "d", "d"], dtype=pl.String),
            pl.Series("value", [2, 7, 374466, 132675], dtype=pl.Int64),
        ],
    )

    assert_eq_df_lf(
        pl.selectors.date(),
        ["b", "a"],
        [
            pl.Series("b", [1, 1], dtype=pl.Int64),
            pl.Series("a", ["x", "z"], dtype=pl.String),
            pl.Series("variable", ["d", "d"], dtype=pl.String),
            pl.Series("value", [day_a, day_b], dtype=pl.Date),
        ],
    )


def test_unpivot_filter_opt() -> None:
    data = {"a": [5, 2, 8, 2], "b": [99, 33, 77, 44]}

    def assert_eq_df_lf(
        on: Any, index: Any, filter_pred: Any, expected_data: list[pl.Series]
    ) -> None:
        def logic(frame: Any, on: Any, index: Any) -> Any:
            return frame.unpivot(on, index=index).filter(filter_pred)

        return assert_eq_df_lf_impl(data, logic, on, index, expected_data)

    assert_eq_df_lf(
        "b",
        "a",
        pl.col.a == 2,
        [
            pl.Series("a", [2, 2], dtype=pl.Int64),
            pl.Series("variable", ["b", "b"], dtype=pl.String),
            pl.Series("value", [33, 44], dtype=pl.Int64),
        ],
    )

    assert_eq_df_lf(
        "b",
        ["b", "a"],
        pl.col.b != 33,
        [
            pl.Series("b", [99, 77, 44], dtype=pl.Int64),
            pl.Series("a", [5, 8, 2], dtype=pl.Int64),
            pl.Series("variable", ["b", "b", "b"], dtype=pl.String),
            pl.Series("value", [99, 77, 44], dtype=pl.Int64),
        ],
    )


def test_unpivot_variable_value_name_25681() -> None:
    schema = {"foo": pl.String, "value": pl.Null}

    q = pl.LazyFrame().unpivot(variable_name="foo")

    assert q.collect_schema() == schema
    assert_frame_equal(q.collect(), pl.DataFrame(schema=schema))


def test_unpivot_projection_pushdown_schema_25720() -> None:
    left = pl.LazyFrame({"date": ["2025-01-01"], "1": [True]})
    right = pl.LazyFrame({"date": ["2025-01-01"], "id": ["1"], "x": [1.0]})

    left_unpivot = left.unpivot(index="date", variable_name="id", value_name="mask")

    q = left_unpivot.join(right, on=["date", "id"], how="left")

    assert q.collect_schema() == {
        "date": pl.String,
        "id": pl.String,
        "mask": pl.Boolean,
        "x": pl.Float64,
    }

    assert_frame_equal(
        q.collect(),
        pl.DataFrame(
            [
                pl.Series("date", ["2025-01-01"], dtype=pl.String),
                pl.Series("id", ["1"], dtype=pl.String),
                pl.Series("mask", [True], dtype=pl.Boolean),
                pl.Series("x", [1.0], dtype=pl.Float64),
            ]
        ),
    )
