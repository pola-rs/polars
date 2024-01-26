from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_group_by(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(eager_execution=True)
    ctx.register("foods", lf)

    out = ctx.execute(
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
    assert out.to_dict(as_series=False) == {
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
    assert ctx.tables() == ["foods"]

    ctx.register("test", lf)
    assert ctx.tables() == ["foods", "test"]

    out = ctx.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(as_series=False) == {"grp": ["c"], "n_dist_attr": [2]}
