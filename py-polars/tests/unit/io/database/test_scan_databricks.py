"""
`scan_database` against a Databricks SQL warehouse.

These run when `POLARS_TEST_DATABRICKS_URI` is set (see `_databricks.py`, which also
seeds the tables). Every query goes over the internet to a metered warehouse and takes
seconds, so each test reads a handful of small tables. The generic matrix (every
row, the predicate corpus, partitions, types) is in `test_scan_servers.py`; here are the
checks that only Databricks needs: its `TIMESTAMP` is a local-time-zone type
(`TIMESTAMP_NTZ` is the naive one), `TIME`, collations in column types, views, and
string literals.
"""

from __future__ import annotations

import contextlib
import os
import re
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.database._databricks import CHAR_VALUES, ENV, ensure_seeded

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

pytest.importorskip("polars_database")
pytest.importorskip("sqlglot")

URI = os.environ.get(ENV)

pytestmark = [
    pytest.mark.skipif(URI is None, reason=f"{ENV} is not set"),
]


def _uri(**params: str) -> str:
    assert URI is not None
    ensure_seeded(URI)
    sep = "&" if "?" in URI else "?"
    return URI + "".join(
        f"{sep if i == 0 else '&'}{k}={v}" for i, (k, v) in enumerate(params.items())
    )


def _connection(**kwargs: Any) -> Callable[[], Any]:
    from tests.unit.io.database._databricks import connect

    assert URI is not None
    ensure_seeded(URI)
    return lambda: connect(URI, **kwargs)


def _scan(source: str, how: str = "adbc", **kwargs: Any) -> pl.LazyFrame:
    """`how`: `adbc`, `databricks.sql` (from the URI), or `factory` (a connection)."""
    if how == "factory":
        return pl.scan_database(source, _connection(), **kwargs)
    return pl.scan_database(source, _uri(), driver=how, **kwargs)


def _filters(lf: pl.LazyFrame) -> tuple[int, int, int]:
    """(pushed, total, exact) filters of the plan."""
    m = re.search(r"filters (\d+)/(\d+) \((\d+) exact\)", lf.explain())
    assert m is not None, lf.explain()
    return int(m[1]), int(m[2]), int(m[3])


def _ids(df: pl.DataFrame) -> list[int]:
    return sorted(df["id"].to_list())


def _check(
    lf: pl.LazyFrame, reference: pl.DataFrame, predicates: dict[str, pl.Expr]
) -> None:
    """Every predicate, pushed, must return the rows Polars selects from the table."""
    failures = []
    for name, p in predicates.items():
        try:
            got = _ids(lf.filter(p).select("id").collect())
        except Exception as e:
            got = [f"{type(e).__name__}: {e}"]  # type: ignore[list-item]
        expected = _ids(reference.filter(p))
        if got != expected:
            failures.append(f"{name}: got {got}, expected {expected}")
    assert not failures, "\n".join(failures)


# --- TIME ----------------------------------------------------------------------------


def test_time_filters_are_pushed_as_time_literals() -> None:
    # sqlglot writes a Spark `TIME` as `TIMESTAMP`, which a `TIME` column can't be
    # compared with
    lf = _scan("edge_types")
    reference = _scan("edge_types", pushdown=False).collect()
    assert reference.schema["t"] == pl.Time
    predicates = {
        "==": pl.col("t") == time(13, 45, 30, 123456),
        "!=": pl.col("t") != time(13, 45, 30, 123456),
        "<": pl.col("t") < time(13, 45, 30, 123457),
        ">": pl.col("t") > time(0, 0),
        ">=": pl.col("t") >= time(0, 0),
        "between": pl.col("t").is_between(time(1), time(23)),
        "is_null": pl.col("t").is_null(),
    }
    _check(lf, reference, predicates)
    assert _filters(lf.filter(predicates["=="])) == (1, 1, 1)
    sql = lf.filter(predicates["=="]).explain()
    assert "`t` = TIME '13:45:30.123456'" in sql


# --- time zones ----------------------------------------------------------------------

# 2024-03-10 02:30 doesn't exist in Los Angeles (the clocks jump from 02:00 to 03:00),
# 2024-11-03 01:30 happens twice; Kolkata is +05:30 all year
ZONES = ["America/Los_Angeles", "Asia/Kolkata"]


def _zone_predicates(zone: str) -> dict[str, pl.Expr]:
    tz = ZoneInfo(zone)
    ntz, ltz = pl.col("ntz"), pl.col("ltz")
    return {
        "ntz == gap 02:30": ntz == datetime(2024, 3, 10, 2, 30),
        "ntz == 03:30": ntz == datetime(2024, 3, 10, 3, 30),
        "ntz == summer noon": ntz == datetime(2024, 7, 1, 12),
        "ntz >= ambiguous 01:30": ntz >= datetime(2024, 11, 3, 1, 30),
        "ntz < winter noon": ntz < datetime(2024, 1, 1, 12),
        "ltz == 03:30 local": ltz == datetime(2024, 3, 10, 3, 30, tzinfo=tz),
        "ltz == summer noon local": ltz == datetime(2024, 7, 1, 12, tzinfo=tz),
        "ltz >= 11-03 01:30 UTC": ltz.dt.convert_time_zone("UTC")
        >= datetime(2024, 11, 3, 1, 30, tzinfo=UTC),
        "ltz is_null": ltz.is_null(),
    }


@pytest.mark.parametrize("zone", ZONES)
@pytest.mark.parametrize("how", ["adbc", "factory"])
def test_timestamps_in_another_session_time_zone(zone: str, how: str) -> None:
    """
    Naive literals are `TIMESTAMP_NTZ`.

    `TIMESTAMP` would convert them in the session's zone, so the skipped 02:30 matched
    03:30 as well.
    """
    if how == "adbc":
        make: Callable[..., pl.LazyFrame] = lambda **kw: pl.scan_database(  # noqa: E731
            "edge_zones", _uri(timezone=zone), **kw
        )
    else:
        factory = _connection(session_configuration={"timezone": zone})
        make = lambda **kw: pl.scan_database("edge_zones", factory, **kw)  # noqa: E731
    lf = make()
    reference = make(pushdown=False).collect()
    # the zone-aware column is in the session's zone; its instants don't change
    assert reference.schema["ltz"] == pl.Datetime("us", zone)
    assert reference.schema["ntz"] == pl.Datetime("us")
    utc = reference.select(pl.col("ltz").dt.convert_time_zone("UTC"))["ltz"].to_list()
    assert utc[1] == datetime(2024, 3, 10, 10, 30, tzinfo=UTC)
    _check(lf, reference, _zone_predicates(zone))
    # the pushed SQL is explicit about both kinds
    assert _filters(lf.filter(pl.col("ntz") == datetime(2024, 3, 10, 2, 30))) == (
        1,
        1,
        1,
    )


def test_naive_literal_on_a_column_of_unknown_kind_is_not_pushed() -> None:
    # a user who types a `TIMESTAMP` (zone-aware) column as naive: its kind is not NTZ
    schema_overrides = {"ltz": pl.Datetime("us")}
    lf = _scan("edge_zones", schema_overrides=schema_overrides)
    reference = _scan(
        "edge_zones", pushdown=False, schema_overrides=schema_overrides
    ).collect()
    p = pl.col("ltz") == datetime(2024, 3, 10, 10, 30)
    assert _filters(lf.filter(p))[0] == 0
    _check(lf, reference, {"ltz == naive": p})


def test_query_timestamp_columns_are_classified_by_the_result_type() -> None:
    lf = _scan("SELECT id, ntz, ltz FROM edge_zones")
    reference = _scan("SELECT id, ntz, ltz FROM edge_zones", pushdown=False).collect()
    assert _filters(lf.filter(pl.col("ntz") == datetime(2024, 3, 10, 2, 30))) == (
        1,
        1,
        1,
    )
    _check(lf, reference, _zone_predicates("UTC"))


# --- views ---------------------------------------------------------------------------


def test_partitioning_a_view_over_rand_is_refused() -> None:
    # each partition would call `rand()` again and get other rows
    for source in ("rand_view", "SELECT * FROM rand_view"):
        with pytest.raises(pl.exceptions.InvalidOperationError, match="rand_view"):
            _scan(source, partition_on="id", partition_num=4).collect()
    # without an explicit partition column, one read
    assert _scan("rand_view", partition_on=None).collect().height == 100


def test_partitioning_a_view_of_constant_rows() -> None:
    lf = _scan("plain_view", partition_on="id", partition_num=4)
    assert "partitions: 4" in lf.explain()
    out = lf.collect().sort("id")
    assert out["id"].to_list() == list(range(1, 101))
    assert out["twice"].to_list() == list(range(2, 201, 2))


def test_view_through_a_python_driver_is_inspected_too() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="rand_view"):
        _scan(
            "rand_view", "databricks.sql", partition_on="id", partition_num=2
        ).collect()


def test_nested_view_resolves_in_its_creation_namespace() -> None:
    # `nested_plain_view` reads `plain_view` unqualified, as of the schema it was
    # created in: from a session in another schema it is still found (the view's own
    # namespace is the context of its names, not the session's), so it is not "volatile"
    from tests.unit.io.database._databricks import connect, parts

    assert URI is not None
    ensure_seeded(URI)
    other = parts(URI)["catalog"]

    def factory() -> Any:
        return connect(URI, catalog=other, schema="information_schema")

    p = parts(URI)
    qualified = f"{p['catalog']}.{p['schema']}"
    lf = pl.scan_database(
        f"{qualified}.nested_plain_view", factory, partition_on="id", partition_num=4
    )
    assert "partitions: 4" in lf.explain()
    assert lf.collect().height == 100
    # a volatile view behind another is found through the same lookup
    for name in ("nested_rand_view", "nested_rand_query"):
        with pytest.raises(pl.exceptions.InvalidOperationError, match="rand_view"):
            pl.scan_database(
                f"{qualified}.{name}", factory, partition_on="id", partition_num=4
            ).collect()
    # and from a session in the view's schema, unqualified
    with pytest.raises(pl.exceptions.InvalidOperationError, match="rand_view"):
        _scan("nested_rand_view", partition_on="id", partition_num=4).collect()


def test_user_table_functions_make_a_source_volatile() -> None:
    # a SQL table function over `rand()`: each partition would call it again
    from tests.unit.io.database._databricks import parts

    assert URI is not None
    p = parts(URI)
    udtf = f"{p['catalog']}.{p['schema']}.rand_udtf"
    for source in ("udtf_view", f"SELECT * FROM {udtf}()"):
        lf = _scan(source, partition_on="id", partition_num=4)
        with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
            lf.collect()


def test_temporary_views_are_not_inspectable_and_shadow_tables() -> None:
    from tests.unit.io.database._databricks import connect, execute

    assert URI is not None
    ensure_seeded(URI)
    conn = connect(URI)
    try:
        execute(
            conn,
            "CREATE OR REPLACE TEMP VIEW tmp_rand AS "
            "SELECT id, rand() AS r FROM range(1, 101) AS t(id)",
        )
        # a temporary view hides the table `orders` of its name on this connection
        execute(
            conn,
            "CREATE OR REPLACE TEMP VIEW orders AS "
            "SELECT id, rand() AS r FROM range(1, 101) AS t(id)",
        )
        for name in ("tmp_rand", "orders", "system.session.tmp_rand"):
            lf = pl.scan_database(name, conn, partition_on="id", partition_num=4)
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match=r"tmp_rand|orders"
            ):
                lf.collect()
        # the shadowing view is what is read
        assert pl.scan_database("orders", conn).collect_schema().names() == ["id", "r"]
    finally:
        conn.close()
    # another connection sees the table
    assert _scan("orders").collect_schema().names()[:2] == ["id", "customer"]


def test_view_with_schema_evolution_has_the_current_collation() -> None:
    # the stored schema of a `WITH SCHEMA EVOLUTION` view is stale until something reads
    # it; the server compares by the base column's current collation
    from tests.unit.io.database._databricks import connect, execute

    assert URI is not None
    ensure_seeded(URI)
    conn = connect(URI)
    try:
        for stmt in [
            "DROP VIEW IF EXISTS collation_v",
            "DROP TABLE IF EXISTS collation_t",
            "CREATE TABLE collation_t (id INT, s STRING)",
            "INSERT INTO collation_t VALUES (1,'abc'),(2,'ABC'),(3,'Abc'),(4,'abd'),(5,NULL)",
            "CREATE VIEW collation_v WITH SCHEMA EVOLUTION AS SELECT * FROM collation_t",
            "ALTER TABLE collation_t ALTER COLUMN s TYPE STRING COLLATE UTF8_LCASE",
        ]:
            execute(conn, stmt)
        # the first scan, before anything reads the view
        lf = pl.scan_database("collation_v", conn)
        assert _ids(lf.filter(pl.col("s") == "abc").collect()) == [1]
    finally:
        for stmt in [
            "DROP VIEW IF EXISTS collation_v",
            "DROP TABLE IF EXISTS collation_t",
        ]:
            execute(conn, stmt)
        conn.close()


# --- strings -------------------------------------------------------------------------


def _pushable(v: str) -> bool:
    return not any(ord(c) < 32 and c not in "\t\n\r" for c in v)


def test_string_literals_with_control_characters() -> None:
    """Control characters other than tab, newline and carriage return aren't pushed."""
    lf = _scan("edge_chars")
    reference = _scan("edge_chars", pushdown=False).collect()
    assert reference.height == len(CHAR_VALUES) + 1
    failures = []
    for v in CHAR_VALUES:
        p = pl.col("s") == v
        got = _ids(lf.filter(p).select("id").collect())
        expected = _ids(reference.filter(p))
        if got != expected or len(expected) != 1:
            failures.append(f"{v!r}: got {got}, expected {expected}")
        pushed = _filters(lf.filter(p))[0] == 1
        if pushed != _pushable(v):
            failures.append(f"{v!r}: pushed={pushed}")
    assert not failures, "\n".join(failures)


def test_string_predicates_with_special_characters() -> None:
    lf = _scan("edge_chars")
    reference = _scan("edge_chars", pushdown=False).collect()
    s = pl.col("s")
    _check(
        lf,
        reference,
        {
            "contains tab": s.str.contains("\t", literal=True),
            "contains backslash": s.str.contains("\\", literal=True),
            "contains quote": s.str.contains("'", literal=True),
            "contains percent": s.str.contains("%", literal=True),
            "contains underscore": s.str.contains("_", literal=True),
            "starts_with bell": s.str.starts_with("a\x07"),
            "ends_with newline": s.str.ends_with("\n"),
            "is_in": s.is_in(["é", "😀", "it's", "a\x00b", ""]),
            "<": s < "b",
            ">=": s >= "é",
            "!=": s != "plain",
            "~==": ~(s == "tab\there"),
        },
    )


# collation: `edge_collations` columns, which of them compare like Polars, and a table
# whose default collation applies to a column declared without one
COLLATED = {"s": True, "lc": False, "ci": False, "ai": False, "u": False}


def test_collations_are_classified_and_never_lose_rows() -> None:
    reference = _scan("edge_collations", pushdown=False).collect()
    lf = _scan("edge_collations")
    failures = []
    for col, exact in COLLATED.items():
        c = pl.col(col)
        predicates = {
            "==": c == "abc",
            "!=": c != "abc",
            "<": c < "b",
            ">=": c >= "B",
            "is_in": c.is_in(["abc", "É"]),
            "starts_with": c.str.starts_with("a"),
            "~==": ~(c == "ABC"),
            "ends_with": c.str.ends_with("c"),
            "eq é": c == "é",
        }
        try:
            _check(lf, reference, predicates)
        except AssertionError as e:
            failures.append(f"{col}: {e}")
        # only the binary collation's comparison is pushed as exact
        _, _, n_exact = _filters(lf.filter(c == "abc"))
        if (n_exact == 1) != exact:
            failures.append(f"{col}: exact={n_exact}")
    assert not failures, "\n".join(failures)


def test_default_collation_of_a_table_applies_to_its_columns() -> None:
    reference = _scan("edge_default_collation", pushdown=False).collect()
    lf = _scan("edge_default_collation")
    # `s` takes the table's `UTF8_LCASE`: the database equates 'abc' and 'ABC'
    for col, exact in (("s", False), ("b", True)):
        c = pl.col(col)
        _check(
            lf,
            reference,
            {"==": c == "abc", "!=": c != "abc", "~==": ~(c == "ABC"), "<": c < "b"},
        )
        assert (_filters(lf.filter(c == "abc"))[2] == 1) == exact, col
    # through a query: the result type tells
    q = _scan("SELECT id, s, concat(s, 'x') AS c FROM edge_default_collation")
    qref = _scan(
        "SELECT id, s, concat(s, 'x') AS c FROM edge_default_collation", pushdown=False
    ).collect()
    _check(
        q,
        qref,
        {
            "s !=": pl.col("s") != "abc",
            "c ==": pl.col("c") == "abcx",
            "c <": pl.col("c") < "b",
        },
    )
    assert _filters(q.filter(pl.col("s") == "abc"))[2] == 0


def test_char_columns_compare_like_polars() -> None:
    # blank-padded types are classified as not exact; whatever the database does with
    # trailing spaces, the rows are the ones Polars selects
    source = "SELECT 1 AS id, CAST('ab' AS CHAR(5)) AS c, CAST('ab' AS VARCHAR(5)) AS v"
    lf = _scan(source)
    reference = _scan(source, pushdown=False).collect()
    c, v = pl.col("c"), pl.col("v")
    _check(
        lf,
        reference,
        {
            "c == ab": c == "ab",
            "c == padded": c == "ab   ",
            "v == ab": v == "ab",
            "c !=": c != "ab",
        },
    )


def test_narrow_and_unusual_types() -> None:
    schemas = {
        how: dict(_scan("edge_small", how).collect_schema())
        for how in ("adbc", "databricks.sql")
    }
    assert schemas["adbc"] == schemas["databricks.sql"]
    assert schemas["adbc"]["ty"] == pl.Int8
    # a single-precision float keeps its value: a double literal doesn't match it
    assert schemas["adbc"]["fl"] == pl.Float32
    # intervals read as the text Databricks writes
    assert schemas["adbc"]["iv"] == pl.String
    lf = _scan("edge_small")
    reference = _scan("edge_small", pushdown=False).collect()
    assert reference["iv"].to_list()[0] == "1 02:03:04.500000000"
    assert reference["ym"].to_list()[:2] == ["1-2", "-0-1"]
    fl, ty = pl.col("fl"), pl.col("ty")
    ch, vc = pl.col("ch"), pl.col("vc")
    _check(
        lf,
        reference,
        {
            "fl == 1.1": fl == 1.1,
            "fl > 1.1": fl > 1.1,
            "fl <= 0.1": fl <= 0.1,
            "fl == 2**24 + 1": fl == 16777217.0,
            "fl in": fl.is_in([0.1, 1.1]),
            "ty == min": ty == -128,
            "ty < 200": ty < 200,
            "ty == 200": ty == 200,
            "ch == ab": ch == "ab",
            "ch == padded": ch == "ab   ",
            "ch == empty": ch == "",
            "vc == x": vc == "x",
            "vc == 'x '": vc == "x ",
            # Databricks casts the literal to an interval, which matches row 1
            "iv == text": pl.col("iv") == "1 2:3:4.5",
            "iv == written": pl.col("iv") == "1 02:03:04.500000000",
            "ym == text": pl.col("ym") == "1-2",
        },
    )
    # a non-text type read as text is never compared in the database
    for col, text in (("iv", "1 2:3:4.5"), ("ym", "1-2")):
        assert _filters(lf.filter(pl.col(col) == text))[0] == 0
    # text types still are
    assert _filters(lf.filter(pl.col("vc") == "x"))[0] == 1


# --- decimals ------------------------------------------------------------------------

WIDE = "12345678901234567890123456.1234567890"  # DECIMAL(38, 10)


@pytest.mark.parametrize("how", ["adbc", "databricks.sql", "factory"])
def test_wide_decimals_are_exact(how: str) -> None:
    lf = _scan(
        f"SELECT 1 AS id, CAST('{WIDE}' AS DECIMAL(38,10)) AS x, "
        "CAST('-99999999999999999999999999999999999999' AS DECIMAL(38,0)) AS y",
        how,
    )
    out = lf.collect()
    assert out["x"].item() == Decimal(WIDE)
    assert out["y"].item() == Decimal("-99999999999999999999999999999999999999")
    assert lf.filter(pl.col("x") == Decimal(WIDE)).collect().height == 1
    assert lf.filter(pl.col("x") > Decimal(WIDE)).collect().height == 0


@pytest.mark.parametrize("how", ["adbc", "databricks.sql", "factory"])
def test_invalid_utf8_text_is_an_error_not_a_string(how: str) -> None:
    """A STRING of other bytes would be a String that fails Arrow validation."""
    sql = "SELECT CAST(unhex('FF') AS STRING) AS s"

    def read_and_validate() -> None:
        _scan(sql, how).collect().to_arrow().validate(full=True)

    with pytest.raises(Exception, match=r"(?i)utf-?8"):
        read_and_validate()
    # a BINARY of the same byte, and valid text, read
    ok = _scan(
        "SELECT unhex('FF') AS b, CAST(unhex('EFBFBD') AS STRING) AS s", how
    ).collect()
    assert ok["b"].to_list() == [b"\xff"]
    assert ok["s"].to_list() == ["\ufffd"]


def test_decimal_predicates_on_a_table() -> None:
    lf = _scan("edge_types")
    reference = _scan("edge_types", pushdown=False).collect()
    max38 = Decimal("99999999999999999999999999999999999999")
    _check(
        lf,
        reference,
        {
            "d38 > 0": pl.col("d38") > Decimal(0),
            "d38 == max": pl.col("d38") == max38,
            "d38 == -max": pl.col("d38") == Decimal("-" + str(max38)),
            "nu < 0": pl.col("nu") < Decimal(0),
            "nu == 123.456789012345": pl.col("nu") == Decimal("123.456789012345"),
            "nu == finer": pl.col("nu") == Decimal("123.4567890123450001"),
        },
    )


# --- partitions ----------------------------------------------------------------------


def _orders_sums() -> tuple[int, int]:
    from tests.unit.io.database._databricks import connect, execute

    assert URI is not None
    conn = connect(URI)
    try:
        count, total = execute(conn, "SELECT count(*), sum(id) FROM orders")[0]
    finally:
        conn.close()
    return int(count), int(total)


@pytest.mark.parametrize(
    ("how", "partition_on", "partition_num"),
    [
        ("adbc", "id", 1),
        ("adbc", "id", 4),
        ("adbc", "ordered_on", 4),
        ("adbc", "amount", 3),
        ("adbc", "price", 5),
        ("adbc", "auto", None),
        ("databricks.sql", "id", 4),
    ],
)
def test_partitions_read_every_row_once(
    how: str, partition_on: str, partition_num: int | None
) -> None:
    # the counts are the server's own; only the checksums cross the link
    count, total = _orders_sums()
    kwargs: dict[str, Any] = {"partition_on": partition_on}
    if partition_num is not None:
        kwargs["partition_num"] = partition_num
    lf = _scan("orders", how, **kwargs)
    if partition_num is not None:
        assert f"partitions: {partition_num}" in lf.explain()
    out = lf.select(
        pl.len().alias("n"),
        pl.col("id").n_unique().alias("u"),
        pl.col("id").sum().alias("s"),
    ).collect()
    assert out.row(0) == (count, count, total)


def test_query_partitions_read_every_row_once() -> None:
    count, total = _orders_sums()
    lf = _scan("SELECT * FROM orders", partition_on="id", partition_num=4)
    assert "partitions: 4" in lf.explain()
    assert lf.select(pl.len(), pl.col("id").sum()).collect().row(0) == (count, total)


def test_partitions_over_a_filter_and_the_null_range() -> None:
    lf = _scan("orders", partition_on="id", partition_num=4).filter(
        pl.col("customer").is_null()
    )
    nulls = lf.select(pl.len()).collect().item()
    assert nulls == len([i for i in range(1, 5001) if i % 7 == 0])


def test_dates_and_timestamps_of_orders_round_trip() -> None:
    lf = _scan("orders", partition_on=None).filter(pl.col("id").is_in([1, 365, 5000]))
    out = lf.collect().sort("id")
    assert out["ordered_on"].to_list() == [
        date(2024, 1, 2),
        date(2024, 1, 1),
        date(2024, 9, 12),
    ]
    assert out["created_at"].to_list() == [
        datetime(2024, 1, 1, 0, 0, 1),
        datetime(2024, 1, 1, 0, 6, 5),
        datetime(2024, 1, 1, 1, 23, 20),
    ]


# --- parameters and connection forms -------------------------------------------------


def test_adbc_params_are_refused_with_a_hint() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match="binds parameters"):
        _scan("SELECT id FROM edge WHERE id < ?", params=[30]).collect()


@pytest.mark.parametrize(
    ("sql", "params"),
    [
        ("SELECT id FROM edge WHERE id < ?", [30]),
        ("SELECT id FROM edge WHERE id < :n", {"n": 30}),
        ("SELECT id FROM edge WHERE id < %(n)s", {"n": 30}),
    ],
)
def test_params_with_the_python_driver(sql: str, params: Any) -> None:
    lf = _scan(sql, "databricks.sql", params=params, partition_on=None)
    assert _ids(lf.collect()) == list(range(30))


def test_every_connection_form_reads_the_same_frame() -> None:
    frames = [
        _scan("edge_types", how).collect().sort("id")
        for how in ("adbc", "databricks.sql", "factory")
    ]
    for other in frames[1:]:
        assert_frame_equal(frames[0], other)


def test_the_token_is_not_in_the_plan() -> None:
    assert URI is not None
    token = re.search(r"token:([^@]+)@", URI)
    assert token is not None
    assert token[1] not in _scan("edge").explain()


# --- a SQLAlchemy Engine ------------------------------------------------------------


def _engine(zone: str | None = None) -> Any:
    sqlalchemy = pytest.importorskip("sqlalchemy")
    pytest.importorskip("databricks.sqlalchemy")
    from tests.unit.io.database._databricks import parts

    assert URI is not None
    ensure_seeded(URI)
    p = parts(URI)
    connect_args: dict[str, Any] = {}
    if zone is not None:
        connect_args["session_configuration"] = {"timezone": zone}
    return sqlalchemy.create_engine(
        f"databricks://token:{p['token']}@{p['host']}?http_path={p['http_path']}"
        f"&catalog={p['catalog']}&schema={p['schema']}",
        connect_args=connect_args,
    )


_NESTED = """
SELECT 1 AS id,
  array(CAST(9007199254740993 AS BIGINT), NULL) AS ab,
  array(CAST('12345678901234567890.123456789' AS DECIMAL(38,9))) AS adec,
  array(TIMESTAMP '2024-07-01 12:00:00+00:00') AS ats,
  map('k', CAST(9007199254740993 AS BIGINT)) AS mb,
  named_struct('a', CAST(9007199254740993 AS BIGINT),
               'b', TIMESTAMP '2024-07-01 12:00:00+00:00') AS st
"""


@pytest.mark.parametrize("zone", [None, "America/Los_Angeles"])
def test_engine_reads_nested_values_and_timestamps_like_the_connector(
    zone: str | None,
) -> None:
    # the Engine's rows typed a NULL-holding array of bigints Float64 (9007199254740993
    # read as ...992.0), failed on other nested types, and read a TIMESTAMP as naive UTC
    # wall-clock time whatever the session's zone
    kwargs = {"session_configuration": {"timezone": zone}} if zone else {}
    factory = _connection(**kwargs)
    engine = _engine(zone)
    try:
        for source in [_NESTED, "SELECT id, ntz, ltz FROM edge_zones"]:
            reference = pl.scan_database(source, factory, partition_on=None).collect()
            got = pl.scan_database(source, engine, partition_on=None).collect()
            assert got.schema == reference.schema, source
            assert_frame_equal(got, reference)
        nested = pl.scan_database(_NESTED, engine, partition_on=None).collect()
        assert nested["ab"].to_list() == [[9007199254740993, None]]
        assert nested["st"][0]["a"] == 9007199254740993
        zoned = pl.scan_database(
            "SELECT id, ltz FROM edge_zones", engine, partition_on=None
        ).collect()
        if zone is not None:
            assert zoned.schema["ltz"] == pl.Datetime("us", zone)
    finally:
        engine.dispose()


def test_sqlalchemy_connection_reads_on_its_own_session() -> None:
    # the Engine's pooled connection was dropped when the cursor was made (and went back
    # into the pool while in use), and a Connection read on another pooled connection:
    # its temporary view wasn't there
    engine = _engine()
    try:
        with engine.connect() as conn:
            conn.exec_driver_sql(
                "CREATE OR REPLACE TEMPORARY VIEW r42_tmp AS SELECT 7 AS x"
            )
            lf = pl.scan_database("SELECT x FROM r42_tmp", conn, partition_on=None)
            assert lf.collect()["x"].to_list() == [7]
            # the Connection is still the user's, and open
            assert conn.exec_driver_sql("SELECT x FROM r42_tmp").fetchall() == [(7,)]
        pl.scan_database("SELECT 1 AS x", engine, partition_on=None).collect()
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()


def test_engine_read_database_early_stop_gives_back_the_pooled_connection() -> None:
    # a batch iterator dropped after one batch, or failing part-way, left the pooled
    # connection checked out for good (a cycle through the driver's connection)
    engine = _engine()
    try:
        for _ in range(2):
            it = pl.read_database(
                "SELECT id FROM edge_zones", engine, iter_batches=True, batch_size=1
            )
            next(iter(it))
            del it
            assert engine.pool.checkedout() == 0
        it = pl.read_database(
            "SELECT id FROM edge_zones", engine, iter_batches=True, batch_size=1
        )
        gen: Any = iter(it)
        next(gen)
        with pytest.raises(RuntimeError, match="boom"):
            gen.throw(RuntimeError("boom"))
        assert engine.pool.checkedout() == 0
    finally:
        engine.dispose()


def test_temporary_functions_of_the_session_are_found() -> None:
    # a temporary function (`system.session`) of a name sqlglot knows (`to_double`,
    # `datetime`: no Databricks built-ins) runs rand() each time; `SHOW USER FUNCTIONS`
    # lists it, and the SQL path's schemas don't
    from tests.unit.io.database._databricks import connect, execute

    assert URI is not None
    ensure_seeded(URI)
    conn = connect(URI)
    try:
        execute(
            conn,
            "CREATE OR REPLACE TEMPORARY FUNCTION to_double(x DOUBLE) RETURNS DOUBLE "
            "RETURN x + rand()",
        )
        lf = pl.scan_database(
            "SELECT id, to_double(id) AS t FROM range(1, 201) AS t(id)",
            conn,
            partition_on="id",
            partition_num=4,
        )
        with pytest.raises(
            pl.exceptions.InvalidOperationError,
            match=r"this query can't be partitioned.*to_double",
        ):
            lf.collect()
        # a query that calls no such function partitions
        lf = pl.scan_database(
            "SELECT id, upper('a') AS u FROM range(1, 201) AS t(id)",
            conn,
            partition_on="id",
            partition_num=4,
        )
        assert sorted(lf.collect()["id"].to_list()) == list(range(1, 201))
    finally:
        execute(conn, "DROP TEMPORARY FUNCTION IF EXISTS to_double")
        conn.close()


def test_casts_of_columns_to_dates_are_proven() -> None:
    # Databricks reads `'now'` in a cast of a literal only; a column, a CTE's or a
    # VALUES' text fails (CAST_INVALID_INPUT)
    sql = "SELECT id, CAST(d AS DATE) AS k FROM {src}"
    src = (
        "(SELECT id, TIMESTAMP '2024-01-01 10:00:00' AS d FROM range(1, 201) AS t(id))"
    )
    lf = _scan(sql.format(src=src), partition_on="id", partition_num=4)
    assert sorted(lf.collect()["id"].to_list()) == list(range(1, 201))
    lf = _scan(
        "SELECT id, CAST('now' AS DATE) AS k FROM range(1, 201) AS t(id)",
        partition_on="id",
        partition_num=4,
    )
    with pytest.raises(pl.exceptions.InvalidOperationError, match="partitioned"):
        lf.collect()


@contextlib.contextmanager
def _scratch(setup: list[str], *, path: str | None = None) -> Iterator[str]:
    """
    Run `setup` statements in the test schema and yield the qualified schema.

    The objects are dropped by name afterwards (`DROP ... IF EXISTS` of each `CREATE`).
    """
    from tests.unit.io.database._databricks import connect, execute, parts

    assert URI is not None
    ensure_seeded(URI)
    p = parts(URI)
    schema = f"`{p['catalog']}`.`{p['schema']}`"
    conn = connect(URI)
    drops: list[str] = []
    try:
        if path is not None:
            execute(conn, f"SET PATH = {path.format(schema=schema)}")
        for sql in setup:
            sql = sql.format(schema=schema)
            m = re.match(
                r"CREATE (?:OR REPLACE )?(TABLE|VIEW|FUNCTION) ([^\s(]+)", sql, re.I
            )
            assert m is not None, sql
            execute(conn, sql)
            drops.append(f"DROP {m[1]} IF EXISTS {m[2]}")
        if path is not None:
            execute(conn, "SET PATH = DEFAULT_PATH")
        yield schema
    finally:
        if path is not None:
            execute(conn, "SET PATH = DEFAULT_PATH")
        for sql in reversed(drops):
            execute(conn, sql)
        conn.close()


def _refused(lf: pl.LazyFrame, match: str = "partitioned") -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        lf.collect()


def test_date_ranges_at_or_before_the_gregorian_cutover_are_refused() -> None:
    # Databricks reads a date of 1582-10-05..14 back as 1582-10-15 but compares it as
    # written, so a cut at or before 1582-10-15 loses rows
    setup = [
        "CREATE OR REPLACE TABLE {schema}.vol_dates AS SELECT * FROM VALUES "
        "(1, DATE'1500-01-01'), (2, DATE'1582-10-04'), (3, DATE'1582-10-10'), "
        "(4, DATE'1582-10-15'), (5, DATE'1700-06-01'), (6, DATE'2024-01-01') "
        "AS t(id, d)"
    ]
    with _scratch(setup) as schema:
        name = f"{schema}.vol_dates"
        kwargs_options: list[Any] = [
            {"partition_range": (date(1500, 1, 1), date(1700, 1, 1))},
            {"partition_range": (date(1582, 10, 15), date(1700, 1, 1))},
            {},  # MIN/MAX: the table's own first date is 1500
        ]
        for kwargs in kwargs_options:
            lf = _scan(name, partition_on="d", partition_num=3, **kwargs)
            _refused(lf, "1582-10-15")
        # without an explicit key, one query reads every row
        auto = _scan(name, partition_on="auto", partition_num=3)
        assert _ids(auto.collect()) == [1, 2, 3, 4, 5, 6]
        # a range above the gap partitions, and reads every row once
        lf = _scan(
            name,
            partition_on="d",
            partition_num=3,
            partition_range=(date(1600, 1, 1), date(2000, 1, 1)),
        )
        assert "partitions: 3" in lf.explain()
        assert _ids(lf.collect()) == [1, 2, 3, 4, 5, 6]


def test_a_view_with_columns_named_like_the_describe_headings() -> None:
    # columns named `# Detailed Table Information` and `Type` must not be read as the
    # view's details, which would make the view an ordinary table
    setup = [
        "CREATE OR REPLACE VIEW {schema}.vol_headings AS SELECT id, "
        "'x' AS `Type`, rand() AS `# Detailed Table Information` "
        "FROM range(1, 201) AS t(id)"
    ]
    with _scratch(setup) as schema:
        name = f"{schema}.vol_headings"
        for source in (name, f"SELECT * FROM {name}"):
            _refused(_scan(source, partition_on="id", partition_num=4))
        assert _scan(name, partition_on=None).collect().height == 200


def test_function_path_of_a_view_and_of_the_session() -> None:
    # `SET PATH = <schema>, system.builtin` puts a user's `abs()` ahead of the built-in
    # one, for a view created then and for a query of that session
    from tests.unit.io.database._databricks import connect, execute

    udf = (
        "CREATE OR REPLACE FUNCTION {schema}.abs(x BIGINT) RETURNS DOUBLE "
        "NOT DETERMINISTIC RETURN rand()"
    )
    view = (
        "CREATE OR REPLACE VIEW {schema}.vol_path_view AS "
        "SELECT id, abs(id) AS a FROM range(1, 201) AS t(id)"
    )
    plain = (
        "CREATE OR REPLACE VIEW {schema}.vol_plain_view AS "
        "SELECT id, sign(id) AS a FROM range(1, 201) AS t(id)"
    )
    with _scratch([udf]) as schema, _scratch([view], path="{schema}, system.builtin"):
        assert URI is not None
        # the view's own path is read from its description
        conn = connect(URI)
        try:
            rows = execute(conn, f"DESCRIBE TABLE EXTENDED {schema}.vol_path_view")
        finally:
            conn.close()
        paths = [r[1] for r in rows if r[0] == "SQL Path"]
        assert paths
        assert not paths[0].lower().startswith("system.builtin"), paths
        name = f"{schema}.vol_path_view"
        for source in (name, f"SELECT * FROM {name}"):
            _refused(_scan(source, partition_on="id", partition_num=4))

        # a query of a session whose path puts the schema first
        def factory() -> Any:
            conn = connect(URI)
            execute(conn, f"SET PATH = {schema}, system.builtin")
            return conn

        sql = "SELECT id, abs(id) AS a FROM range(1, 201) AS t(id)"
        lf = pl.scan_database(sql, factory, partition_on="id", partition_num=4)
        _refused(lf)
        # (a user's `abs` in a schema of the path counts wherever the path puts it: the
        # catalog lists routines by name, not by what resolves first)
        _refused(_scan(sql, partition_on="id", partition_num=4))
        # another built-in, and a view created under the default path, partition
        lf = _scan(
            "SELECT id, sign(id) AS a FROM range(1, 201) AS t(id)",
            partition_on="id",
            partition_num=4,
        )
        assert "partitions: 4" in lf.explain()
        with _scratch([plain]):
            lf = _scan(f"{schema}.vol_plain_view", partition_on="id", partition_num=4)
            assert "partitions: 4" in lf.explain()
            assert lf.collect().height == 200


def test_bound_time_text_makes_the_query_volatile() -> None:
    # `CAST(:text AS TIMESTAMP)` of 'now' is the statement's clock
    sql = "SELECT id, CAST(:text AS TIMESTAMP) AS ts FROM range(8) AS t(id)"
    for kwargs in (
        {"partition_on": "id", "partition_num": 4},
        {"partition_predicates": ["id < 4", "id >= 4"]},
    ):
        lf = _scan(sql, "databricks.sql", params={"text": "now"}, **kwargs)
        _refused(lf)
    # a fixed time, and one scan of 'now', are fine
    fixed = _scan(
        sql,
        "databricks.sql",
        params={"text": "2024-01-01"},
        partition_on="id",
        partition_num=4,
    )
    assert fixed.collect()["ts"].n_unique() == 1
    one = _scan(sql, "databricks.sql", params={"text": "now"}, partition_on=None)
    assert one.collect()["ts"].n_unique() == 1


def test_bare_current_time_makes_the_query_volatile() -> None:
    # the statement's time, which each partition would read at another moment
    for expr in ("CURRENT_TIME", "CURRENT_TIME()", "CURRENT_TIMESTAMP", "CURRENT_DATE"):
        sql = f"SELECT id, {expr} AS stamp FROM range(8) AS t(id)"
        kwargs_options: list[Any] = [
            {"partition_on": "id", "partition_num": 4},
            {"partition_predicates": ["id < 4", "id >= 4"]},
        ]
        for kwargs in kwargs_options:
            _refused(_scan(sql, **kwargs), "partitioned")


def test_collated_grouping_is_not_partitioned_or_filtered() -> None:
    # a GROUP BY over a column that compares 'A' and 'a' equal returns one of them, and
    # a later statement may return the other
    setup = [
        "CREATE OR REPLACE TABLE {schema}.vol_collated AS SELECT id, "
        "CAST(IF(id % 7 < 3, 'A', 'a') AS STRING COLLATE UTF8_LCASE) AS s "
        "FROM range(0, 4000) AS t(id)",
        "CREATE OR REPLACE TABLE {schema}.vol_binary AS SELECT id, "
        "IF(id % 7 < 3, 'A', 'a') AS s FROM range(0, 4000) AS t(id)",
    ]
    with _scratch(setup) as schema:
        grouped = "SELECT id % 32 AS g, ascii(s) AS k FROM {t} GROUP BY id % 32, s"
        sql = grouped.format(t=f"{schema}.vol_collated")
        _refused(_scan(sql, partition_on="k", partition_num=4))
        _refused(_scan(sql, partition_predicates=["k = 65", "k = 97"]))
        lf = _scan(sql, partition_on=None)
        # the database's own representative is read once; a filter on it stays in
        # Polars, so it can't change which spelling is returned
        assert _filters(lf.filter(pl.col("k") == 97))[0] == 0
        assert lf.collect().height > 0
        # a collation in the query's own text
        inline = (
            "SELECT ascii(s) AS k FROM (SELECT CAST(x AS STRING COLLATE UTF8_LCASE) "
            "AS s FROM VALUES ('A'), ('a') AS t(x) GROUP BY ALL)"
        )
        _refused(_scan(inline, partition_on="k", partition_num=2))
        # binary strings group exactly: nothing changes
        sql = grouped.format(t=f"{schema}.vol_binary")
        lf = _scan(sql, partition_on="k", partition_num=2)
        assert "partitions: 2" in lf.explain()
        assert lf.collect().height == _scan(sql, partition_on=None).collect().height


def test_user_function_named_like_another_dialects_builtin() -> None:
    # `total` is SQLite's, not Databricks': a user's function of the name in the schema
    # of the path runs, whatever sqlglot calls it
    udf = (
        "CREATE OR REPLACE FUNCTION {schema}.total(x BIGINT) RETURNS DOUBLE "
        "NOT DETERMINISTIC RETURN rand()"
    )
    sql = "SELECT id, total(id) AS t FROM range(1, 201) AS t(id)"
    with _scratch([udf]):
        lf = _scan(sql, partition_on="id", partition_num=4)
        with pytest.raises(
            pl.exceptions.InvalidOperationError,
            match=r"this query can't be partitioned.*total",
        ):
            lf.collect()


def test_a_column_mask_makes_a_table_unproven() -> None:
    # a mask's function (`rand()`) runs on each read, whatever the key: the plan
    # of a masked table is a secure view, which neither DESCRIBE nor the information
    # schema shows for a policy (ABAC)
    from tests.unit.io.database._databricks import connect, execute, parts

    assert URI is not None
    ensure_seeded(URI)
    p = parts(URI)
    q = f"{p['catalog']}.{p['schema']}"
    conn = connect(URI)
    try:
        execute(
            conn,
            f"CREATE OR REPLACE TABLE {q}.vol_masked AS "
            "SELECT CAST(id AS INT) AS id, CAST(id AS INT) AS k FROM range(0, 60)",
        )
        execute(
            conn,
            f"CREATE OR REPLACE FUNCTION {q}.vol_mask(k INT) RETURNS INT "
            "RETURN CAST(floor(rand() * 60) AS INT)",
        )
        execute(
            conn,
            f"ALTER TABLE {q}.vol_masked ALTER COLUMN k SET MASK {q}.vol_mask",
        )
        variants: list[dict[str, Any]] = [
            {"partition_on": "k", "partition_num": 4},
            {"partition_on": "id", "partition_num": 4},
        ]
        for kwargs in variants:
            with pytest.raises(
                pl.exceptions.InvalidOperationError, match="can't be partitioned"
            ):
                _scan(f"{q}.vol_masked", **kwargs).collect()
        # one read still works
        assert _scan(f"{q}.vol_masked", partition_on=None).collect().height == 60
        # a plain table partitions
        execute(
            conn,
            f"CREATE OR REPLACE TABLE {q}.vol_unmasked AS "
            "SELECT CAST(id AS INT) AS id, CAST(id AS INT) AS k FROM range(0, 60)",
        )
        lf = _scan(f"{q}.vol_unmasked", partition_on="k", partition_num=4)
        assert sorted(lf.collect()["id"].to_list()) == list(range(60))
    finally:
        execute(conn, f"DROP TABLE IF EXISTS {q}.vol_masked")
        execute(conn, f"DROP TABLE IF EXISTS {q}.vol_unmasked")
        execute(conn, f"DROP FUNCTION IF EXISTS {q}.vol_mask")
        conn.close()
