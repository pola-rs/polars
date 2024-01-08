from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_case_when() -> None:
    lf = pl.LazyFrame(
        {
            "v1": [None, 2, None, 4],
            "v2": [101, 202, 303, 404],
        }
    )
    with pl.SQLContext(test_data=lf, eager_execution=True) as ctx:
        out = ctx.execute(
            """
            SELECT *, CASE WHEN COALESCE(v1, v2) % 2 != 0 THEN 'odd' ELSE 'even' END as "v3"
            FROM test_data
            """
        )
    assert out.to_dict(as_series=False) == {
        "v1": [None, 2, None, 4],
        "v2": [101, 202, 303, 404],
        "v3": ["odd", "even", "odd", "even"],
    }


def test_control_flow(foods_ipc_path: Path) -> None:
    nums = pl.LazyFrame(
        {
            "x": [1, None, 2, 3, None, 4],
            "y": [5, 4, None, 3, None, 2],
            "z": [3, 4, None, 3, 6, None],
        }
    )
    res = pl.SQLContext(df=nums).execute(
        """
        SELECT
          COALESCE(x,y,z) as "coalsc",
          NULLIF(x, y) as "nullif x_y",
          NULLIF(y, z) as "nullif y_z",
          IFNULL(x, y) as "ifnull x_y",
          IFNULL(y,-1) as "inullf y_z",
          COALESCE(x, NULLIF(y,z)) as "both",
          IF(x = y, 'eq', 'ne') as "x_eq_y",
        FROM df
        """,
        eager=True,
    )

    assert res.to_dict(as_series=False) == {
        "coalsc": [1, 4, 2, 3, 6, 4],
        "nullif x_y": [1, None, 2, None, None, 4],
        "nullif y_z": [5, None, None, None, None, 2],
        "ifnull x_y": [1, 4, 2, 3, None, 4],
        "inullf y_z": [5, 4, -1, 3, -1, 2],
        "both": [1, None, 2, 3, None, 4],
        "x_eq_y": ["ne", "ne", "ne", "eq", "ne", "ne"],
    }
    for null_func in ("IFNULL", "NULLIF"):
        # both functions expect only 2 arguments
        with pytest.raises(InvalidOperationError):
            pl.SQLContext(df=nums).execute(f"SELECT {null_func}(x,y,z) FROM df")
