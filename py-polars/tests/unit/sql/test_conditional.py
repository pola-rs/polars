from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from polars.testing import assert_frame_equal


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
    with pl.SQLContext(test_data=lf, eager=True) as ctx:
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
        with pytest.raises(
            SQLSyntaxError,
            match=r"(IFNULL|NULLIF) expects 2 arguments \(found 3\)",
        ):
            pl.SQLContext(df=nums).execute(f"SELECT {null_func}(x,y,z) FROM df")


def test_greatest_least() -> None:
    df = pl.DataFrame(
        {
            "a": [-100, None, 200, 99],
            "b": [None, -0.1, 99.0, 100.0],
            "c": ["bb", "aa", "dd", "cc"],
            "d": ["cc", "bb", "aa", "dd"],
            "e": [date(1969, 12, 31), date(2021, 1, 2), None, date(2021, 1, 4)],
            "f": [date(1970, 1, 1), date(2000, 10, 20), date(2077, 7, 5), None],
        }
    )
    with pl.SQLContext(df=df) as ctx:
        df_max_horizontal = ctx.execute(
            """
            SELECT
              GREATEST("a", 0, "b") AS max_ab_zero,
              GREATEST("a", "b") AS max_ab,
              GREATEST("c", "d", ) AS max_cd,
              GREATEST("e", "f") AS max_ef,
              GREATEST('1999-12-31'::date, "e", "f") AS max_efx
            FROM df
            """
        ).collect()

        assert_frame_equal(
            df_max_horizontal,
            pl.DataFrame(
                {
                    "max_ab_zero": [0.0, 0.0, 200.0, 100.0],
                    "max_ab": [-100.0, -0.1, 200.0, 100.0],
                    "max_cd": ["cc", "bb", "dd", "dd"],
                    "max_ef": [
                        date(1970, 1, 1),
                        date(2021, 1, 2),
                        date(2077, 7, 5),
                        date(2021, 1, 4),
                    ],
                    "max_efx": [
                        date(1999, 12, 31),
                        date(2021, 1, 2),
                        date(2077, 7, 5),
                        date(2021, 1, 4),
                    ],
                }
            ),
        )

        df_min_horizontal = ctx.execute(
            """
            SELECT
              LEAST("b", "a", 0) AS min_ab_zero,
              LEAST("a", "b") AS min_ab,
              LEAST("c", "d") AS min_cd,
              LEAST("e", "f") AS min_ef,
              LEAST("f", "e", '1999-12-31'::date) AS min_efx
            FROM df
            """
        ).collect()

        assert_frame_equal(
            df_min_horizontal,
            pl.DataFrame(
                {
                    "min_ab_zero": [-100.0, -0.1, 0.0, 0.0],
                    "min_ab": [-100.0, -0.1, 99.0, 99.0],
                    "min_cd": ["bb", "aa", "aa", "cc"],
                    "min_ef": [
                        date(1969, 12, 31),
                        date(2000, 10, 20),
                        date(2077, 7, 5),
                        date(2021, 1, 4),
                    ],
                    "min_efx": [
                        date(1969, 12, 31),
                        date(1999, 12, 31),
                        date(1999, 12, 31),
                        date(1999, 12, 31),
                    ],
                }
            ),
        )
