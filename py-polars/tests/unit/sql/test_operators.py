from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_div() -> None:
    df = pl.LazyFrame(
        {
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [-100.5, 7.0, 2.5, None, -3.14],
        }
    )
    with pl.SQLContext(df=df, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              a / b AS a_div_b,
              a // b AS a_floordiv_b,
              SIGN(b) AS b_sign,
            FROM df
            """
        )

    assert_frame_equal(
        pl.DataFrame(
            [
                [-0.0995024875621891, 2.85714285714286, 12.0, None, -15.92356687898089],
                [-1, 2, 12, None, -16],
                [-1, 1, 1, None, -1],
            ],
            schema=["a_div_b", "a_floordiv_b", "b_sign"],
        ),
        res,
    )


def test_equal_not_equal() -> None:
    # validate null-aware/unaware equality operators
    df = pl.DataFrame({"a": [1, None, 3, 6, 5], "b": [1, None, 3, 4, None]})

    with pl.SQLContext(frame_data=df) as ctx:
        out = ctx.execute(
            """
            SELECT
              -- not null-aware
              (a = b)  as "1_eq_unaware",
              (a <> b) as "2_neq_unaware",
              (a != b) as "3_neq_unaware",
              -- null-aware
              (a <=> b) as "4_eq_aware",
              (a IS NOT DISTINCT FROM b) as "5_eq_aware",
              (a IS DISTINCT FROM b) as "6_neq_aware",
            FROM frame_data
            """
        ).collect()

    assert out.select(cs.contains("_aware").null_count().sum()).row(0) == (0, 0, 0)
    assert out.select(cs.contains("_unaware").null_count().sum()).row(0) == (2, 2, 2)

    assert out.to_dict(as_series=False) == {
        "1_eq_unaware": [True, None, True, False, None],
        "2_neq_unaware": [False, None, False, True, None],
        "3_neq_unaware": [False, None, False, True, None],
        "4_eq_aware": [True, True, True, False, False],
        "5_eq_aware": [True, True, True, False, False],
        "6_neq_aware": [False, False, False, True, True],
    }


def test_is_between(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(foods1=lf, eager=True)
    out = ctx.execute(
        """
        SELECT *
        FROM foods1
        WHERE foods1.calories BETWEEN 22 AND 30
        ORDER BY "calories" DESC, "sugars_g" DESC
    """
    )
    assert out.rows() == [
        ("fruit", 30, 0.0, 5),
        ("vegetables", 30, 0.0, 5),
        ("fruit", 30, 0.0, 3),
        ("vegetables", 25, 0.0, 4),
        ("vegetables", 25, 0.0, 3),
        ("vegetables", 25, 0.0, 2),
        ("vegetables", 22, 0.0, 3),
    ]
    out = ctx.execute(
        """
        SELECT *
        FROM foods1
        WHERE calories NOT BETWEEN 22 AND 30
        ORDER BY "calories" ASC
        """
    )
    assert not any((22 <= cal <= 30) for cal in out["calories"])


def test_starts_with() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["aaa", "bbb", "a"],
            "y": ["abc", "b", "aa"],
        },
    )
    assert lf.sql("SELECT x ^@ 'a' AS x_starts_with_a FROM self").collect().rows() == [
        (True,),
        (False,),
        (True,),
    ]
    assert lf.sql("SELECT x ^@ y AS x_starts_with_y FROM self").collect().rows() == [
        (False,),
        (True,),
        (False,),
    ]


@pytest.mark.parametrize("match_float", [False, True])
def test_unary_ops_8890(match_float: bool) -> None:
    with pl.SQLContext(
        df=pl.DataFrame({"a": [-2, -1, 1, 2], "b": ["w", "x", "y", "z"]}),
    ) as ctx:
        in_values = "(-3.0, -1.0, +2.0, +4.0)" if match_float else "(-3, -1, +2, +4)"
        res = ctx.execute(
            f"""
            SELECT *, -(3) as c, (+4) as d
            FROM df WHERE a IN {in_values}
            """
        )
        assert res.collect().to_dict(as_series=False) == {
            "a": [-1, 2],
            "b": ["x", "z"],
            "c": [-3, -3],
            "d": [4, 4],
        }
