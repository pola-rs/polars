from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from tests.unit.sql.asserts import assert_sql_matches


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_order_by_basic(foods_ipc_path: Path) -> None:
    foods = pl.scan_ipc(foods_ipc_path)

    order_by_distinct_res = foods.sql(
        """
        SELECT DISTINCT category
        FROM self
        ORDER BY category DESC
        """
    ).collect()
    assert order_by_distinct_res.to_dict(as_series=False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"]
    }

    for category in ("category", "category AS cat"):
        category_col = category.split(" ")[-1]
        order_by_group_by_res = foods.sql(
            f"""
            SELECT {category}
            FROM self
            GROUP BY category
            ORDER BY {category_col} DESC
            """
        ).collect()
        assert order_by_group_by_res.to_dict(as_series=False) == {
            category_col: ["vegetables", "seafood", "meat", "fruit"]
        }

    order_by_constructed_group_by_res = foods.sql(
        """
        SELECT category, SUM(calories) as summed_calories
        FROM self
        GROUP BY category
        ORDER BY summed_calories DESC
        """
    ).collect()
    assert order_by_constructed_group_by_res.to_dict(as_series=False) == {
        "category": ["seafood", "meat", "fruit", "vegetables"],
        "summed_calories": [1250, 540, 410, 192],
    }

    order_by_unselected_res = foods.sql(
        """
        SELECT SUM(calories) as summed_calories
        FROM self
        GROUP BY category
        ORDER BY summed_calories DESC
        """
    ).collect()
    assert order_by_unselected_res.to_dict(as_series=False) == {
        "summed_calories": [1250, 540, 410, 192],
    }


def test_order_by_misc_selection() -> None:
    df = pl.DataFrame({"x": [None, 1, 2, 3], "y": [4, 2, None, 8]})

    # order by aliased col
    res = df.sql("SELECT x, y AS y2 FROM self ORDER BY y2")
    assert res.to_dict(as_series=False) == {"x": [1, None, 3, 2], "y2": [2, 4, 8, None]}

    res = df.sql("SELECT x, y AS y2 FROM self ORDER BY y2 DESC")
    assert res.to_dict(as_series=False) == {"x": [2, 3, None, 1], "y2": [None, 8, 4, 2]}

    # order by col found in wildcard
    res = df.sql("SELECT *, y AS y2 FROM self ORDER BY y")
    assert res.to_dict(as_series=False) == {
        "x": [1, None, 3, 2],
        "y": [2, 4, 8, None],
        "y2": [2, 4, 8, None],
    }
    res = df.sql("SELECT *, y AS y2 FROM self ORDER BY y NULLS FIRST")
    assert res.to_dict(as_series=False) == {
        "x": [2, 1, None, 3],
        "y": [None, 2, 4, 8],
        "y2": [None, 2, 4, 8],
    }

    # order by col found in qualified wildcard
    res = df.sql("SELECT self.* FROM self ORDER BY x NULLS FIRST")
    assert res.to_dict(as_series=False) == {"x": [None, 1, 2, 3], "y": [4, 2, None, 8]}

    res = df.sql("SELECT self.* FROM self ORDER BY y NULLS FIRST")
    assert res.to_dict(as_series=False) == {"x": [2, 1, None, 3], "y": [None, 2, 4, 8]}

    # order by col excluded from wildcard
    res = df.sql("SELECT * EXCLUDE y FROM self ORDER BY y")
    assert res.to_dict(as_series=False) == {"x": [1, None, 3, 2]}

    res = df.sql("SELECT * EXCLUDE y FROM self ORDER BY y NULLS FIRST")
    assert res.to_dict(as_series=False) == {"x": [2, 1, None, 3]}

    # order by col excluded from qualified wildcard
    res = df.sql("SELECT self.* EXCLUDE y FROM self ORDER BY y")
    assert res.to_dict(as_series=False) == {"x": [1, None, 3, 2]}

    # order by expression
    res = df.sql("SELECT (x % y) AS xmy FROM self ORDER BY -(x % y)")
    assert res.to_dict(as_series=False) == {"xmy": [3, 1, None, None]}

    res = df.sql("SELECT (x % y) AS xmy FROM self ORDER BY x % y NULLS FIRST")
    assert res.to_dict(as_series=False) == {"xmy": [None, None, 1, 3]}

    # confirm that 'order by all' syntax prioritises cols
    df = pl.DataFrame({"SOME": [0, 1], "ALL": [1, 0]})
    res = df.sql("SELECT * FROM self ORDER BY ALL")
    assert res.to_dict(as_series=False) == {"SOME": [1, 0], "ALL": [0, 1]}

    res = df.sql("SELECT * FROM self ORDER BY ALL DESC")
    assert res.to_dict(as_series=False) == {"SOME": [0, 1], "ALL": [1, 0]}


def test_order_by_misc_16579() -> None:
    res = pl.DataFrame(
        {
            "x": ["apple", "orange"],
            "y": ["sheep", "alligator"],
            "z": ["hello", "world"],
        }
    ).sql(
        """
        SELECT z, y, x
        FROM self ORDER BY y DESC
        """
    )
    assert res.columns == ["z", "y", "x"]
    assert res.to_dict(as_series=False) == {
        "z": ["hello", "world"],
        "y": ["sheep", "alligator"],
        "x": ["apple", "orange"],
    }


def test_order_by_multi_nulls_first_last() -> None:
    df = pl.DataFrame({"x": [None, 1, None, 3], "y": [3, 2, None, 1]})
    # ┌──────┬──────┐
    # │ x    ┆ y    │
    # │ ---  ┆ ---  │
    # │ i64  ┆ i64  │
    # ╞══════╪══════╡
    # │ null ┆ 3    │
    # │ 1    ┆ 2    │
    # │ null ┆ null │
    # │ 3    ┆ 1    │
    # └──────┴──────┘

    res1 = df.sql("SELECT * FROM self ORDER BY x, y")
    res2 = df.sql("SELECT * FROM self ORDER BY ALL")
    for res in (res1, res2):
        assert res.to_dict(as_series=False) == {
            "x": [1, 3, None, None],
            "y": [2, 1, 3, None],
        }

    res = df.sql("SELECT * FROM self ORDER BY x NULLS FIRST, y")
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 1, 3],
        "y": [3, None, 2, 1],
    }

    res = df.sql("SELECT * FROM self ORDER BY x, y NULLS FIRST")
    assert res.to_dict(as_series=False) == {
        "x": [1, 3, None, None],
        "y": [2, 1, None, 3],
    }

    res1 = df.sql("SELECT * FROM self ORDER BY x NULLS FIRST, y NULLS FIRST")
    res2 = df.sql("SELECT * FROM self ORDER BY All NULLS FIRST")
    for res in (res1, res2):
        assert res.to_dict(as_series=False) == {
            "x": [None, None, 1, 3],
            "y": [None, 3, 2, 1],
        }

    res1 = df.sql("SELECT * FROM self ORDER BY x DESC NULLS FIRST, y DESC NULLS FIRST")
    res2 = df.sql("SELECT * FROM self ORDER BY all DESC NULLS FIRST")
    for res in (res1, res2):
        assert res.to_dict(as_series=False) == {
            "x": [None, None, 3, 1],
            "y": [None, 3, 1, 2],
        }

    res = df.sql("SELECT * FROM self ORDER BY x DESC NULLS FIRST, y DESC NULLS LAST")
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 3, 1],
        "y": [3, None, 1, 2],
    }

    res = df.sql("SELECT * FROM self ORDER BY y DESC NULLS FIRST, x NULLS LAST")
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 1, 3],
        "y": [None, 3, 2, 1],
    }


def test_order_by_ordinal() -> None:
    df = pl.DataFrame({"x": [None, 1, None, 3], "y": [3, 2, None, 1]})

    res = df.sql("SELECT * FROM self ORDER BY 1, 2")
    assert res.to_dict(as_series=False) == {
        "x": [1, 3, None, None],
        "y": [2, 1, 3, None],
    }

    res = df.sql("SELECT * FROM self ORDER BY 1 DESC, 2")
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 3, 1],
        "y": [3, None, 1, 2],
    }

    res = df.sql("SELECT * FROM self ORDER BY 1 DESC NULLS LAST, 2 ASC")
    assert res.to_dict(as_series=False) == {
        "x": [3, 1, None, None],
        "y": [1, 2, 3, None],
    }

    res = df.sql("SELECT * FROM self ORDER BY 1 DESC NULLS LAST, 2 ASC NULLS FIRST")
    assert res.to_dict(as_series=False) == {
        "x": [3, 1, None, None],
        "y": [1, 2, None, 3],
    }

    res = df.sql("SELECT * FROM self ORDER BY 1 DESC, 2 DESC NULLS FIRST")
    assert res.to_dict(as_series=False) == {
        "x": [None, None, 3, 1],
        "y": [None, 3, 1, 2],
    }


def test_order_by_errors() -> None:
    df = pl.DataFrame({"a": ["w", "x", "y", "z"], "b": [1, 2, 3, 4]})

    with pytest.raises(
        SQLInterfaceError,
        match="ORDER BY ordinal value must refer to a valid column; found 99",
    ):
        df.sql("SELECT * FROM self ORDER BY 99")

    with pytest.raises(
        SQLSyntaxError,
        match="negative ordinal values are invalid for ORDER BY; found -1",
    ):
        df.sql("SELECT * FROM self ORDER BY -1")


@pytest.mark.parametrize(
    "query",
    [
        # basic aliasing: ORDER BY original column name after aliasing
        "SELECT a b FROM self GROUP BY a ORDER BY a",
        "SELECT a AS b FROM self GROUP BY a ORDER BY a",
        # table-qualified aliasing
        "SELECT self.a b FROM self GROUP BY self.a ORDER BY self.a",
        "SELECT self.a AS b FROM self GROUP BY self.a ORDER BY self.a",
        # mixed qualified/unqualified
        "SELECT a b FROM self GROUP BY a ORDER BY self.a",
        "SELECT self.a b FROM self GROUP BY a ORDER BY a",
        # ORDER BY alias name (should still work)
        "SELECT a b FROM self GROUP BY a ORDER BY b",
        "SELECT a AS b FROM self GROUP BY a ORDER BY b",
    ],
)
def test_order_by_aliased_group_key(query: str) -> None:
    """Test ORDER BY with original column name when aliased in SELECT."""
    df = pl.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
    assert_sql_matches(df, query=query, compare_with="sqlite")


@pytest.mark.parametrize(
    "query",
    [
        # cross-aliasing: columns swap names
        "SELECT a AS b, b AS a FROM self GROUP BY a, b ORDER BY self.a",
        "SELECT a AS b, b AS a FROM self GROUP BY a, b ORDER BY self.b",
        # cross-aliasing with expressions
        "SELECT a AS b, -b AS a FROM self GROUP BY a, b ORDER BY self.a",
        "SELECT a AS b, -b AS a FROM self GROUP BY a, b ORDER BY self.b",
        # cross-aliasing with aggregate
        "SELECT a AS b, SUM(b) AS a FROM self GROUP BY a ORDER BY self.a",
        "SELECT a AS b, SUM(b) AS a FROM self GROUP BY a ORDER BY self.b",
    ],
)
def test_order_by_cross_aliased_columns(query: str) -> None:
    """Test ORDER BY with cross-aliasing where columns swap names."""
    df = pl.DataFrame({"a": [3, 1, 2], "b": [30, 10, 20]})
    assert_sql_matches(df, query=query, compare_with="sqlite")


@pytest.mark.parametrize(
    "query",
    [
        # multiple columns with various aliasing patterns
        "SELECT a x, b y, a + b z FROM self GROUP BY a, b ORDER BY a",
        "SELECT a x, b y, a + b z FROM self GROUP BY a, b ORDER BY self.a",
        "SELECT a x, b y, a + b z FROM self GROUP BY a, b ORDER BY b DESC",
        "SELECT a x, b y, a + b z FROM self GROUP BY a, b ORDER BY self.b DESC",
        # ORDER BY referencing original columns
        "SELECT a x, b y FROM self GROUP BY a, b ORDER BY a + b",
        "SELECT a x, b y FROM self GROUP BY a, b ORDER BY self.a + self.b",
        # ORDER BY with ordinal
        "SELECT a x, b y FROM self GROUP BY a, b ORDER BY 1",
        "SELECT a x, b y FROM self GROUP BY a, b ORDER BY 2 DESC",
    ],
)
def test_order_by_multi_column_aliasing(query: str) -> None:
    """Test ORDER BY with multiple aliased columns and expressions."""
    df = pl.DataFrame({"a": [3, 1, 2, 4], "b": [30, 10, 20, 15]})
    assert_sql_matches(df, query=query, compare_with="sqlite")


@pytest.mark.parametrize(
    "query",
    [
        # aggregate with aliased group key
        "SELECT a grp, SUM(b) total FROM self GROUP BY a ORDER BY a",
        "SELECT a grp, SUM(b) total FROM self GROUP BY a ORDER BY self.a",
        "SELECT a grp, SUM(b) total FROM self GROUP BY a ORDER BY grp",
        "SELECT a grp, SUM(b) total FROM self GROUP BY a ORDER BY total DESC",
        # multiple aggregates
        "SELECT a grp, SUM(b) s, AVG(b) avg FROM self GROUP BY a ORDER BY a",
        "SELECT a grp, SUM(b) s, AVG(b) avg FROM self GROUP BY a ORDER BY self.a DESC",
    ],
)
def test_order_by_aggregate_with_aliased_key(query: str) -> None:
    """Test ORDER BY with aggregates and aliased group keys."""
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": [10, 20, 30, 40, 50]})
    assert_sql_matches(df, query=query, compare_with="sqlite")
