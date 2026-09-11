from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


def test_sql_plan_roundtrip() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    q = lf.sql("SELECT b, a * 2 AS a2 FROM self WHERE a > 1")

    roundtripped = pl.LazyFrame.deserialize(q.serialize())
    assert_frame_equal(roundtripped.collect(), q.collect())


def test_sql_query_is_stored_unexpanded() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    query = "SELECT a FROM self WHERE a > 1"

    assert query.encode() in lf.sql(query).serialize()


def test_sql_join_roundtrip() -> None:
    lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": [6, 7, 8]})
    lf2 = pl.LazyFrame({"a": [3, 2, 1], "d": [125, -654, 888]})

    q = pl.sql("SELECT lf1.a, b, d FROM lf1 INNER JOIN lf2 USING (a) ORDER BY a")
    roundtripped = pl.LazyFrame.deserialize(q.serialize())

    assert roundtripped.collect().to_dict(as_series=False) == {
        "a": [1, 2, 3],
        "b": [6, 7, 8],
        "d": [888, -654, 125],
    }


def test_sql_schema_resolved_at_collect(tmp_path: Path) -> None:
    # the wildcard is expanded against the data as it is at collect time; the
    # serialized plan carries the query, not a schema frozen when it was built
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": [1, 2]}).write_parquet(path)

    serialized = pl.scan_parquet(path).sql("SELECT * FROM self").serialize()
    assert pl.LazyFrame.deserialize(serialized).collect().columns == ["a"]

    pl.DataFrame({"a": [1, 2], "b": [3, 4]}).write_parquet(path)
    assert pl.LazyFrame.deserialize(serialized).collect().columns == ["a", "b"]


def test_sql_errors_surface_at_collect() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    q = lf.sql("SELECT does_not_exist FROM self")

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        q.collect()


def test_sql_snapshots_registered_tables() -> None:
    ctx = pl.SQLContext(tbl=pl.LazyFrame({"a": [1, 2, 3]}))
    q = ctx.execute("SELECT a FROM tbl")
    ctx.unregister("tbl")

    assert q.collect().to_series().to_list() == [1, 2, 3]


def test_sql_unknown_table_errors_at_collect() -> None:
    ctx = pl.SQLContext()
    q = ctx.execute("SELECT * FROM nope")

    with pytest.raises(SQLInterfaceError, match="relation 'nope' was not found"):
        q.collect()


def test_sql_defers_with_case_insensitive_table_name() -> None:
    ctx = pl.SQLContext()
    ctx.register("MyTbl", pl.LazyFrame({"a": [1, 2, 3]}))

    assert ctx.execute("SELECT a FROM mytbl").collect().to_series().to_list() == [
        1,
        2,
        3,
    ]


def test_unnest_alias_stays_registered() -> None:
    with pl.SQLContext(eager=True) as ctx:
        assert ctx.execute(
            "SELECT x FROM UNNEST([1,2,3]) AS u(x)"
        ).to_series().to_list() == [
            1,
            2,
            3,
        ]
        assert "u" in ctx.tables()


def test_table_function_stays_registered(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a\n1\n2\n")

    with pl.SQLContext(eager=True) as ctx:
        ctx.execute(f"SELECT a FROM read_csv('{path.as_posix()}')")

        # the name a table function registers under is the path it was given, whose
        # spelling is platform-dependent; only the registration itself matters here
        tables = ctx.tables()
        assert len(tables) == 1
        assert tables[0].endswith("data.csv")


def test_quantified_subquery_survives_roundtrip() -> None:
    # the statement is desugared before being cached on the node, so the cached and the
    # re-parsed path must agree
    t1 = pl.LazyFrame({"a": [1, 5, 9], "b": [2, 5, 6]})

    q = pl.sql(
        "SELECT a FROM t1 o WHERE b = ANY (SELECT x.b FROM t1 AS x WHERE x.a = o.a) "
        "ORDER BY a"
    )
    expected = [1, 5, 9]

    assert q.collect().to_series().to_list() == expected
    assert (
        pl.LazyFrame.deserialize(q.serialize()).collect().to_series().to_list()
        == expected
    )


def test_collect_schema_then_collect_agree() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    q = lf.sql("SELECT b, a * 2 AS d FROM self WHERE a > 1")

    assert q.collect_schema().names() == ["b", "d"]
    assert q.collect().to_dict(as_series=False) == {"b": ["y", "z"], "d": [4, 6]}
