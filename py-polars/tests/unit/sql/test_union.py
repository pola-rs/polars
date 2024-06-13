from __future__ import annotations

import pytest

import polars as pl


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
