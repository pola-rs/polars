from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal


def test_except_intersect() -> None:
    df1 = pl.DataFrame({"x": [1, 9, 1, 1], "y": [2, 3, 4, 4], "z": [5, 5, 5, 5]})  # noqa: F841
    df2 = pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4], "z": [7, 6, 5]})  # noqa: F841

    res_e = pl.sql("SELECT x, y, z FROM df1 EXCEPT SELECT * FROM df2", eager=True)
    res_i = pl.sql("SELECT * FROM df1 INTERSECT SELECT x, y, z FROM df2", eager=True)

    assert sorted(res_e.rows()) == [(1, 2, 5), (9, 3, 5)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]

    res_e = pl.sql("SELECT * FROM df2 EXCEPT TABLE df1", eager=True)
    res_i = pl.sql(
        """
        SELECT * FROM df2
        INTERSECT
        SELECT x::int8, y::int8, z::int8
          FROM (VALUES (1,2,5),(9,3,5),(1,4,5),(1,4,5)) AS df1(x,y,z)
        """,
        eager=True,
    )
    assert sorted(res_e.rows()) == [(1, 2, 7), (9, None, 6)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]

    # check null behaviour of nulls
    with pl.SQLContext(
        tbl1=pl.DataFrame({"x": [2, 9, 1], "y": [2, None, 4]}),
        tbl2=pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4]}),
    ) as ctx:
        res = ctx.execute("SELECT * FROM tbl1 EXCEPT SELECT * FROM tbl2", eager=True)
        assert_frame_equal(pl.DataFrame({"x": [2], "y": [2]}), res)


def test_except_intersect_by_name() -> None:
    df1 = pl.DataFrame(  # noqa: F841
        {
            "x": [1, 9, 1, 1],
            "y": [2, 3, 4, 4],
            "z": [5, 5, 5, 5],
        }
    )
    df2 = pl.DataFrame(  # noqa: F841
        {
            "y": [2, None, 4],
            "w": ["?", "!", "%"],
            "z": [7, 6, 5],
            "x": [1, 9, 1],
        }
    )
    res_e = pl.sql(
        "SELECT x, y, z FROM df1 EXCEPT BY NAME SELECT * FROM df2",
        eager=True,
    )
    res_i = pl.sql(
        "SELECT * FROM df1 INTERSECT BY NAME SELECT * FROM df2",
        eager=True,
    )
    assert sorted(res_e.rows()) == [(1, 2, 5), (9, 3, 5)]
    assert sorted(res_i.rows()) == [(1, 4, 5)]
    assert res_e.columns == ["x", "y", "z"]
    assert res_i.columns == ["x", "y", "z"]


@pytest.mark.parametrize(
    ("op", "op_subtype"),
    [
        ("EXCEPT", "ALL"),
        ("EXCEPT", "ALL BY NAME"),
        ("INTERSECT", "ALL"),
        ("INTERSECT", "ALL BY NAME"),
    ],
)
def test_except_intersect_all_unsupported(op: str, op_subtype: str) -> None:
    df1 = pl.DataFrame({"n": [1, 1, 1, 2, 2, 2, 3]})  # noqa: F841
    df2 = pl.DataFrame({"n": [1, 1, 2, 2]})  # noqa: F841

    with pytest.raises(
        SQLInterfaceError,
        match=f"'{op} {op_subtype}' is not supported",
    ):
        pl.sql(f"SELECT * FROM df1 {op} {op_subtype} SELECT * FROM df2")


@pytest.mark.parametrize("op", ["EXCEPT", "INTERSECT", "UNION"])
def test_except_intersect_errors(op: str) -> None:
    df1 = pl.DataFrame({"x": [1, 9, 1, 1], "y": [2, 3, 4, 4], "z": [5, 5, 5, 5]})  # noqa: F841
    df2 = pl.DataFrame({"x": [1, 9, 1], "y": [2, None, 4], "z": [7, 6, 5]})  # noqa: F841

    if op != "UNION":
        with pytest.raises(
            SQLInterfaceError,
            match=f"'{op} ALL' is not supported",
        ):
            pl.sql(f"SELECT * FROM df1 {op} ALL SELECT * FROM df2", eager=False)

    with pytest.raises(
        SQLInterfaceError,
        match=f"{op} requires equal number of columns in each table",
    ):
        pl.sql(f"SELECT x FROM df1 {op} SELECT x, y FROM df2", eager=False)


@pytest.mark.parametrize(
    ("cols1", "cols2", "union_subtype", "expected"),
    [
        (
            ["*"],
            ["*"],
            "",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["frame2.*"],
            "ALL",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["frame1.*"],
            ["c1", "c2"],
            "DISTINCT",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["c2", "c1"],
            "ALL BY NAME",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["c1", "c2"],
            ["c2", "c1"],
            "BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        pytest.param(
            ["c1", "c2"],
            ["c2", "c1"],
            "DISTINCT BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
    ],
)
def test_union(
    cols1: list[str],
    cols2: list[str],
    union_subtype: str,
    expected: list[tuple[int, str]],
) -> None:
    with pl.SQLContext(
        frame1=pl.DataFrame({"c1": [1, 2], "c2": ["zz", "yy"]}),
        frame2=pl.DataFrame({"c1": [2, 3], "c2": ["yy", "xx"]}),
        eager=True,
    ) as ctx:
        query = f"""
            SELECT {', '.join(cols1)} FROM frame1
            UNION {union_subtype}
            SELECT {', '.join(cols2)} FROM frame2
        """
        assert sorted(ctx.execute(query).rows()) == expected
