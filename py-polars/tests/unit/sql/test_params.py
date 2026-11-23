from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    SQLInterfaceError,
    SQLSyntaxError,
)
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["a", "b", "c", "d"],
            "dt": [date(2024, 1, d) for d in (1, 2, 3, 4)],
        }
    )


@pytest.mark.parametrize(
    ("query", "params"),
    [
        ("SELECT name FROM self WHERE id > ? AND id < ?", [1, 4]),
        ("SELECT name FROM self WHERE id > $1 AND id < $2", [1, 4]),
        ("SELECT name FROM self WHERE id < $2 AND id > $1", (1, 4)),
        ("SELECT name FROM self WHERE id > ?1 AND id < ?2", [1, 4]),
        ("SELECT name FROM self WHERE id > $lo AND id < $hi", {"lo": 1, "hi": 4}),
        ("SELECT name FROM self WHERE id > :lo AND id < :hi", {"hi": 4, "lo": 1}),
        ("SELECT name FROM self WHERE id > $lo AND id < :hi", {"lo": 1, "hi": 4}),
    ],
)
def test_params_placeholders(
    df: pl.DataFrame, query: str, params: list[Any] | dict[str, Any]
) -> None:
    expected = pl.DataFrame({"name": ["b", "c"]})
    assert_frame_equal(df.sql(query, params=params), expected)
    assert_frame_equal(df.lazy().sql(query, params=params).collect(), expected)
    ctx = pl.SQLContext(frames={"self": df})
    assert_frame_equal(ctx.execute(query, params=params, eager=True), expected)
    assert_frame_equal(
        pl.sql(query.replace("self", "df"), params=params, eager=True), expected
    )


def test_params_values() -> None:
    # an int has the type of the equal SQL literal
    values = [
        None,
        True,
        42,
        -(2**63),
        2**64 - 1,
        1.5,
        float("inf"),
        "it's",
        b"\x00\xff",
        date(2024, 2, 29),
        datetime(2024, 2, 29, 12, 30, 15, 123456),
        datetime(2024, 2, 29, 12, 30, tzinfo=UTC),
        timedelta(days=1, microseconds=5),
        Decimal("123.450"),
        [1, 2, 3],
    ]
    query = "SELECT " + ", ".join(f"${i + 1} AS c{i}" for i in range(len(values)))
    res = pl.sql(query, params=values, eager=True)
    assert res.row(0) == tuple(values)
    assert res.dtypes == [
        pl.Null,
        pl.Boolean,
        pl.Int32,
        pl.Int64,
        pl.Int128,
        pl.Float64,
        pl.Float64,
        pl.String,
        pl.Binary,
        pl.Date,
        pl.Datetime("us"),
        pl.Datetime("us", "UTC"),
        pl.Duration("us"),
        pl.Decimal(38, 3),
        pl.List(pl.Int64),
    ]


def test_params_are_values(df: pl.DataFrame) -> None:
    # a parameter is never SQL text
    for name in ["a' OR '1'='1", "a; DROP TABLE self", "name"]:
        res = df.sql("SELECT id FROM self WHERE name = $1", params=[name])
        assert res.height == 0

    res = df.sql("SELECT id FROM self WHERE name IN (?, ?)", params=["b", "d"])
    assert res["id"].to_list() == [2, 4]

    # a negative number acts as a negated literal
    res = pl.sql(
        "SELECT $1 AS a, -$1 AS b, ARRAY[-$1, $2] AS c, ARRAY[-$3] AS d",
        params=[-5, 3, Decimal("1.5")],
        eager=True,
    )
    assert res.row(0) == (-5, 5, [5, 3], [Decimal("-1.5")])
    assert res.dtypes[:2] == pl.sql("SELECT -5 AS a, 5 AS b", eager=True).dtypes

    # one parameter read twice
    res = df.sql("SELECT $1 + id AS x FROM self WHERE id <= $1", params=[2])
    assert res["x"].to_list() == [3, 4]


def test_params_in_literal_positions(df: pl.DataFrame, tmp_path: Path) -> None:
    res = df.sql("SELECT id FROM self ORDER BY id LIMIT $1 OFFSET $2", params=[2, 1])
    assert res["id"].to_list() == [2, 3]

    res = df.sql(
        "SELECT STRFTIME(dt, $fmt) AS s FROM self WHERE dt > $d",
        params={"fmt": "%d/%m", "d": date(2024, 1, 3)},
    )
    assert res["s"].to_list() == ["04/01"]

    res = df.sql(
        "SELECT dt + INTERVAL $1 AS dt FROM self WHERE id = 1", params=["2 days"]
    )
    assert res["dt"].to_list() == [date(2024, 1, 3)]

    res = df.sql("SELECT DATE_PART($1, dt) AS p FROM self WHERE id = 2", params=["day"])
    assert res["p"].to_list() == [2]

    res = pl.sql("SELECT TRIM(BOTH ? FROM ?) AS s", params=["x", "xxaxx"], eager=True)
    assert res["s"].to_list() == ["a"]

    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,NA\n2,x\n")
    res = pl.sql(
        "SELECT * FROM read_csv($1, null_values => $2, schema_overrides => $3)",
        params=[str(path), ["NA"], {"a": "SMALLINT"}],
        eager=True,
    )
    assert res.rows() == [(1, None), (2, "x")]
    assert res.dtypes == [pl.Int16, pl.String]

    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    res = pl.sql(
        "SELECT name FROM read_parquet($path) WHERE id = $id",
        params={"path": str(path), "id": 3},
        eager=True,
    )
    assert res["name"].to_list() == ["c"]


def test_params_in_subqueries(df: pl.DataFrame) -> None:
    res = df.sql(
        """
        WITH t AS (SELECT * FROM self WHERE id >= $lo)
        SELECT name FROM t
        WHERE dt IN (SELECT dt FROM self WHERE id <= $hi)
          AND EXISTS (SELECT 1 FROM self s WHERE s.id = t.id + $step)
        ORDER BY name
        """,
        params={"lo": 2, "hi": 3, "step": 1},
    )
    assert res["name"].to_list() == ["b", "c"]

    res = df.sql(
        "SELECT id, COUNT(*) OVER () AS n FROM self GROUP BY id HAVING id > $1",
        params=[2],
    )
    assert res.sort("id").rows() == [(3, 2), (4, 2)]

    # values without a SQL literal form
    res = df.sql(
        """
        SELECT id, (SELECT MAX(dt) FROM self WHERE dt < $d) AS m FROM self s
        WHERE dt IN (SELECT dt FROM self WHERE dt >= $d)
          AND EXISTS (SELECT 1 FROM self t WHERE t.id = s.id AND t.dt < $end)
          AND id IN (SELECT x FROM UNNEST($ids) AS u(x))
        """,
        params={"d": date(2024, 1, 2), "end": date(2024, 1, 4), "ids": [1, 2, 3]},
    )
    assert res.sort("id").rows() == [(2, date(2024, 1, 1)), (3, date(2024, 1, 1))]


def test_params_nested_and_numpy() -> None:
    import numpy as np

    res = pl.sql(
        "SELECT $1 AS l, $2 AS s, $3 AND TRUE AS b, $4 + 1 AS i",
        params=[[1, None], {"x": "a", "y": [1.5]}, np.bool_(True), np.int8(5)],
        eager=True,
    )
    assert res.row(0) == ([1, None], {"x": "a", "y": [1.5]}, True, 6)

    # negating the smallest Int64 widens
    df = pl.DataFrame({"x": [-(2**63), 0]})
    res = df.sql("SELECT x FROM self WHERE x IN (-$1)", params=[-(2**63)])
    assert res.height == 0
    res = pl.sql("SELECT ARRAY[-$1] AS a", params=[-(2**63)], eager=True)
    assert res.item().to_list() == [2**63]
    res = pl.sql("SELECT -$1 AS a", params=[-(2**63)], eager=True)
    assert res.item() == 2**63
    for value in [-(2**127), 2**128 - 1]:
        with pytest.raises(
            ComputeError, match=r"negating query parameter \$1 overflows"
        ):
            pl.sql("SELECT -$1 AS a", params=[value])
        with pytest.raises(
            ComputeError, match=r"negating query parameter \$1 overflows"
        ):
            pl.sql("SELECT -(+($1)) AS a", params=[value])


def test_params_serialize(df: pl.DataFrame) -> None:
    lf = df.lazy().sql("SELECT id FROM self WHERE dt > $1", params=[date(2024, 1, 2)])
    lf = pl.LazyFrame.deserialize(lf.serialize())
    assert lf.collect()["id"].to_list() == [3, 4]


@pytest.mark.parametrize(
    ("query", "params", "error", "match"),
    [
        ("SELECT $1, $2", [1], SQLInterfaceError, r"no value given for query parameter \$2"),
        ("SELECT ?, ?", [1], SQLInterfaceError, r"query has 2 '\?' parameters, but 1 values"),
        ("SELECT $1", [1, 2], SQLInterfaceError, r"query parameter \$2 is not used"),
        ("SELECT $a", {"a": 1, "c": 2, "b": 3}, SQLInterfaceError, r"query parameter \$b is not used"),
        ("SELECT 1", [1], SQLInterfaceError, r"query parameter \$1 is not used"),
        ("SELECT $a", {"b": 1}, SQLInterfaceError, r"no value given for query parameter \$a"),
        ("SELECT $1", None, SQLInterfaceError, r"no value given for query parameter \$1"),
        ("SELECT ?", None, SQLInterfaceError, r"query has 1 '\?' parameters, but 0 values"),
        ("SELECT ?, $1", [1], SQLSyntaxError, r"cannot mix '\?' with numbered"),
        ("SELECT $a", [1], SQLInterfaceError, r"\$a is named, but positional values"),
        ("SELECT :1", {"a": 1}, SQLInterfaceError, r":1 is positional, but named values"),
        ("SELECT $0", [1], SQLSyntaxError, r"numbering starts at 1"),
        ("SELECT * FROM self ORDER BY $1", [1], SQLSyntaxError, r"\$1 cannot be an ORDER BY item"),
        ("SELECT id FROM self GROUP BY ?", [1], SQLSyntaxError, r"\? cannot be a GROUP BY item"),
        ("SELECT id FROM self GROUP BY ROLLUP(id, $1)", [1], SQLSyntaxError, r"\$1 cannot be a GROUP BY item"),
        ("SELECT id FROM self GROUP BY GROUPING SETS ((?), ())", [1], SQLSyntaxError, r"\? cannot be a GROUP BY item"),
        ("SELECT id FROM self OFFSET $1", [-1], InvalidOperationError, r"-1"),
        ("CREATE VIEW v AS SELECT * FROM self WHERE id = $1", [1], SQLInterfaceError, "not supported in CREATE VIEW"),
    ],
)  # fmt: skip
def test_params_errors(
    df: pl.DataFrame,
    query: str,
    params: list[Any] | dict[str, Any] | None,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        pl.SQLContext(frames={"self": df}).execute(query, params=params, eager=True)


def test_params_invalid_values() -> None:
    with pytest.raises(TypeError, match="must be a sequence or a mapping"):
        pl.sql("SELECT $1", params="x")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be scalars, got 'Series'"):
        pl.sql("SELECT $1", params=[pl.Series([1, 2])])
    with pytest.raises(TypeError, match="must be scalars, got 'Expr'"):
        pl.sql("SELECT $x", params={"x": pl.lit(1)})
