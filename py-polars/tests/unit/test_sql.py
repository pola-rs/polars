import os
from pathlib import Path

import pytest

import polars as pl


# TODO: Do not rely on I/O for these tests
@pytest.fixture()
def foods_ipc_path() -> str:
    return str(Path(os.path.dirname(__file__)) / "io" / "files" / "foods1.ipc")


def test_sql_distinct() -> None:
    lf = pl.LazyFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    c = pl.SQLContext()
    c.register("df", lf)
    out = c.query(
        """
        SELECT DISTINCT a
        FROM df
        ORDER BY a DESC
        """
    )
    assert out.to_dict(False) == {"a": [3, 2, 1]}

    out = c.query(
        """
        SELECT DISTINCT
          a*2 AS two_a,
          b/2 AS half_b
        FROM df
        ORDER BY two_a ASC, half_b DESC
        """
    )
    assert out.to_dict(False) == {
        "two_a": [2, 2, 4, 6],
        "half_b": [1, 0, 2, 3],
    }


def test_sql_groupby(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register("foods", lf)

    out = c.query(
        """
        SELECT
            category,
            count(category) as n,
            max(calories),
            min(fats_g)
        FROM foods
        GROUP BY category
        HAVING n > 5
        ORDER BY n, category DESC
        """
    )
    assert out.to_dict(False) == {
        "category": ["vegetables", "fruit", "seafood"],
        "n": [7, 7, 8],
        "calories": [45, 130, 200],
        "fats_g": [0.0, 0.0, 1.5],
    }

    lf = pl.LazyFrame(
        {
            "grp": ["a", "b", "c", "c", "b"],
            "att": ["x", "y", "x", "y", "y"],
        }
    )
    with pytest.raises(TypeError, match="Cannot register.*DataFrame.*use LazyFrame"):
        c.register("test", lf.collect())  # type: ignore[arg-type]

    c.register("test", lf)
    out = c.query(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_sql_join_inner(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register_many(foods1=lf, foods2=lf)

    for join_clause in (
        "ON foods1.category = foods2.category",
        "USING (category)",
    ):
        out = c.query(
            f"""
            SELECT *
            FROM foods1
            INNER JOIN foods2 {join_clause}
            LIMIT 2
            """
        )
        assert out.to_dict(False) == {
            "category": ["vegetables", "vegetables"],
            "calories": [45, 20],
            "fats_g": [0.5, 0.0],
            "sugars_g": [2, 2],
            "calories_right": [45, 45],
            "fats_g_right": [0.5, 0.5],
            "sugars_g_right": [2, 2],
        }


def test_sql_join_left() -> None:
    frames = {
        "tbl_a": pl.LazyFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.LazyFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.LazyFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    c = pl.SQLContext()
    c.register_many(frames)

    out = c.execute(
        """
        SELECT a, b, c, d
        FROM tbl_a
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        ORDER BY c DESC
        """
    )
    assert out.collect().rows() == [
        (1, 4, "z", 25.5),
        (2, None, "y", -50.0),
        (3, 6, "x", None),
    ]


def test_sql_is_between(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register("foods1", lf)

    out = c.query(
        """
        SELECT *
        FROM foods1
        WHERE foods1.calories BETWEEN 20 AND 31
        LIMIT 4
    """
    )
    assert out.to_dict(False) == {
        "category": ["fruit", "vegetables", "fruit", "vegetables"],
        "calories": [30, 25, 30, 22],
        "fats_g": [0.0, 0.0, 0.0, 0.0],
        "sugars_g": [5, 2, 3, 3],
    }

    out = c.query(
        """
        SELECT *
        FROM foods1
        WHERE calories NOT BETWEEN 20 AND 31
        LIMIT 4
        """
    )
    assert out.to_dict(False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"],
        "calories": [45, 150, 100, 60],
        "fats_g": [0.5, 5.0, 5.0, 0.0],
        "sugars_g": [2, 0, 0, 11],
    }


def test_sql_trim(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register("foods1", lf)

    out = c.query(
        """
        SELECT TRIM(LEADING 'v' FROM category)
        FROM foods1
        LIMIT 2
        """
    )
    assert out.to_dict(False) == {"category": ["egetables", "seafood"]}
