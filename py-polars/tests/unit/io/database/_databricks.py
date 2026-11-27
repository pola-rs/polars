"""
The Databricks test schema of the `scan_database` server tests: connection and seeding.

The server tests (`test_scan_servers.py`, `test_scan_databricks.py`) run against a real
workspace when `POLARS_TEST_DATABRICKS_URI` names one, as the ADBC driver's URI:

    databricks://token:<PAT>@<workspace host>:443/sql/1.0/warehouses/<id>
        ?catalog=<catalog>&schema=<schema>

The schema is the tests' own: `ensure_seeded` (re)creates every table in it, and nothing
else is touched; the URI must name both catalog and schema (seeding refuses otherwise),
and every statement names them. Each table is one statement (`CREATE OR REPLACE TABLE
... AS SELECT ... FROM VALUES`), so seeders racing each other (CI jobs) leave whole
tables, and `_seed_version` is written last. Tables stay small (it is a metered
warehouse). Seed by hand with

    python -m tests.unit.io.database._databricks   # from py-polars/

Tables:
- `edge`: the edge-case table of `_scan_corpus.py` (`sb` has an explicit binary
  collation, `ts` is a `TIMESTAMP_NTZ`).
- `orders`: 5,000 rows, generated on the server.
- `edge_types`: types beyond the corpus (decimals up to precision 38, `TIMESTAMP` and
  `TIMESTAMP_NTZ`, `TIME`, `BINARY`, nested types, `VARIANT`).
- `edge_small`: narrow and unusual types (`TINYINT`, `FLOAT`, intervals, `CHAR`).
- `edge_zones`: timestamps around daylight saving transitions, as `TIMESTAMP_NTZ` and
  `TIMESTAMP`.
- `edge_default_collation`: a table whose default collation is `UTF8_LCASE` (`s`;
  `b` is binary).
- `edge_chars`: strings with escapes, control characters, quotes and non-ASCII text.
- `edge_collations`: strings under the collations Databricks has.
- `rand_view`, `plain_view`: views over `rand()` and over constant rows;
  `nested_plain_view`: a view over `plain_view`, naming it unqualified.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import sys
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from tests.unit.io.database._scan_corpus import edge_rows

try:
    import fcntl
except ImportError:  # Windows: no lock, the tests are skipped there anyway
    fcntl = None  # type: ignore[assignment]

ENV = "POLARS_TEST_DATABRICKS_URI"

# bump when the tables below change: a seeded schema of another version is reseeded
VERSION = "2026-11-16.7"

# the rows of `edge_types` (ids 1 and 3 hold values, 2 only NULLs)
_TYPE_ROWS_SQL = """
(1, '{"a": 1, "b": [1, 2]}', X'0001ff', CAST('123.456789012345' AS DECIMAL(30,12)),
 CAST('99999999999999999999999999999999999999' AS DECIMAL(38,0)),
 CAST('2026-03-29 01:30:00.123456+00:00' AS TIMESTAMP), TIMESTAMP_NTZ '2026-03-29 01:30:00.123456',
 TIME '13:45:30.123456', array(1, 2, NULL), named_struct('a', 1, 'b', 'x'),
 map('k', 1, 'l', NULL), parse_json('{"a": [1, 2]}')),
(2, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL),
(3, '[]', X'', CAST('-0.5' AS DECIMAL(30,12)),
 CAST('-99999999999999999999999999999999999999' AS DECIMAL(38,0)),
 CAST('1970-01-02 00:00:00+00:00' AS TIMESTAMP), TIMESTAMP_NTZ '1970-01-02 00:00:00',
 TIME '00:00:00', array(), named_struct('a', NULL, 'b', NULL), map(), parse_json('null'))
"""


def parts(uri: str | None = None) -> dict[str, Any]:
    """The workspace host, HTTP path, token, catalog and schema of a Databricks URI."""
    uri = uri or os.environ[ENV]
    p = urlsplit(uri)
    q = {k: v[-1] for k, v in parse_qs(p.query).items()}
    return {
        "host": p.hostname,
        "port": p.port or 443,
        "http_path": unquote(p.path),
        "token": unquote(p.password or ""),
        "catalog": q.get("catalog"),
        "schema": q.get("schema"),
    }


def connect(uri: str | None = None, **kwargs: Any) -> Any:
    """A `databricks.sql` connection to the test schema."""
    from databricks import sql  # type: ignore[import-not-found]

    p = parts(uri)
    kw: dict[str, Any] = {
        "server_hostname": p["host"],
        "http_path": p["http_path"],
        "access_token": p["token"],
    }
    if p["catalog"]:
        kw["catalog"] = p["catalog"]
    if p["schema"]:
        kw["schema"] = p["schema"]
    return sql.connect(**{**kw, **kwargs})


def execute(conn: Any, statement: str) -> list[Any]:
    """Runs a statement and returns its rows (none for DDL)."""
    with conn.cursor() as cur:
        cur.execute(statement)
        try:
            return list(cur.fetchall())
        except Exception:
            return []


def _quote(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("'", "\\'")
    escaped = escaped.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{escaped}'"


def lit(v: Any) -> str:
    """A Python value as a Databricks SQL literal."""
    from datetime import date, datetime

    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"CAST('{v}' AS BIGINT)"
    if isinstance(v, float):
        if math.isnan(v):
            return "CAST('NaN' AS DOUBLE)"
        if math.isinf(v):
            return f"CAST('{'-' if v < 0 else ''}Infinity' AS DOUBLE)"
        return f"CAST('{v!r}' AS DOUBLE)"
    if isinstance(v, Decimal):
        return f"CAST('{v}' AS DECIMAL(18,4))"
    if isinstance(v, datetime):
        return f"TIMESTAMP_NTZ '{v.isoformat(sep=' ')}'"
    if isinstance(v, date):
        return f"DATE '{v.isoformat()}'"
    if isinstance(v, str):
        # string literals take backslash escapes; other control characters are made by
        # `chr` (a raw one may not survive the statement's transport)
        parts: list[str] = []
        run = ""
        for ch in v:
            if ord(ch) < 32 and ch not in "\t\n\r":
                parts += [_quote(run), f"chr({ord(ch)})"]
                run = ""
            else:
                run += ch
        parts.append(_quote(run))
        parts = [p for p in parts if p != "''"] or ["''"]
        return parts[0] if len(parts) == 1 else f"concat({', '.join(parts)})"
    msg = f"no SQL literal for {v!r}"
    raise TypeError(msg)


# strings that stress literals: escapes, control characters (a NUL before digits is an
# octal escape in some renderers), quotes, non-ASCII
CHAR_VALUES = [
    "plain",
    "tab\there",
    "new\nline",
    "car\rriage",
    "back\\slash",
    "it's",
    'say "hi"',
    "\\n is not a newline",
    "é",
    "😀",
    "a\x07b",
    "a\x0bb",
    "a\x0cb",
    "afb",
    "a\x00b",
    "a\x0012",
    "a\n2",
    "a\x1bb",
    "100%_",
    "",
    " ",
]


def _values(rows: list[tuple[Any, ...]], names: list[str]) -> str:
    """`FROM VALUES (…), … AS v(names)` of Python rows."""
    body = ",\n".join("(" + ", ".join(lit(v) for v in r) + ")" for r in rows)
    return f"FROM VALUES {body} AS v({', '.join(names)})"


def _select(columns: dict[str, str], source: str) -> str:
    """`SELECT CAST(name AS type) AS name, …` of the `FROM …` clause `source`."""
    cols = ", ".join(
        f"CAST({n} AS {t}) AS {n}" if t else f"{n}" for n, t in columns.items()
    )
    return f"SELECT {cols} {source}"


def _target(uri: str | None = None) -> tuple[str, str]:
    """The catalog and schema of the URI; seeding refuses to run without both."""
    p = parts(uri)
    if not p["catalog"] or not p["schema"]:
        msg = (
            f"{ENV} must name the test catalog and schema (?catalog=…&schema=…): "
            "seeding replaces tables, and must not touch the warehouse's default schema"
        )
        raise ValueError(msg)
    return p["catalog"], p["schema"]


def _ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def _refuse_foreign_schema(conn: Any, catalog: str, schema: str) -> None:
    """Seeding replaces tables by name: refused in a schema with others' tables."""
    rows = execute(conn, f"SHOW TABLES IN {_ident(catalog)}.{_ident(schema)}")
    names = {str(r[1]) for r in rows if not (len(r) > 2 and r[2])}
    if names and "_seed_version" not in names:
        msg = (
            f"{catalog}.{schema} holds tables the tests didn't make ({sorted(names)[:5]}); "
            f"seeding replaces tables by name: point {ENV} at an empty schema"
        )
        raise RuntimeError(msg)


def seed(conn: Any, catalog: str, schema: str) -> None:
    """(Re)creates every table and view of the test schema (idempotent)."""
    _refuse_foreign_schema(conn, catalog, schema)

    def t(name: str) -> str:
        return f"{_ident(catalog)}.{_ident(schema)}.{_ident(name)}"

    def run(statement: str) -> None:
        execute(conn, statement)

    edge_columns = {
        "id": "BIGINT",
        "i16": "SMALLINT",
        "i32": "INT",
        "i64": "BIGINT",
        "f64": "DOUBLE",
        "dec": "DECIMAL(18,4)",
        "s": "STRING",
        "sb": "STRING COLLATE UTF8_BINARY",
        "d": "DATE",
        "ts": "TIMESTAMP_NTZ",
        "b": "BOOLEAN",
    }
    edge = _values(edge_rows(nan=True), list(edge_columns))
    run(f"CREATE OR REPLACE TABLE {t('edge')} AS {_select(edge_columns, edge)}")
    run(
        f"""CREATE OR REPLACE TABLE {t("orders")} AS SELECT
            id,
            CASE WHEN id % 7 = 0 THEN NULL ELSE concat('cust_', CAST(id % 100 AS STRING)) END
                AS customer,
            CAST((id % 1000) / 10 AS DECIMAL(12,2)) AS amount,
            CAST(id * 0.5 AS DOUBLE) AS price,
            id % 2 = 0 AS is_paid,
            date_add(DATE '2024-01-01', CAST(id % 365 AS INT)) AS ordered_on,
            timestampadd(SECOND, CAST(id AS INT), TIMESTAMP_NTZ '2024-01-01 00:00:00') AS created_at
            FROM range(1, 5001) AS t(id)"""
    )
    types_columns = {
        "id": "BIGINT",
        "j": "STRING",
        "bin": "BINARY",
        "nu": "DECIMAL(30,12)",
        "d38": "DECIMAL(38,0)",
        "tstz": "TIMESTAMP",
        "tntz": "TIMESTAMP_NTZ",
        "t": "TIME(6)",
        "arr": "ARRAY<INT>",
        "st": "STRUCT<a: INT, b: STRING>",
        "mp": "MAP<STRING, INT>",
        "v": "VARIANT",
    }
    run(
        f"CREATE OR REPLACE TABLE {t('edge_types')} AS "
        + _select(
            types_columns,
            f"FROM VALUES {_TYPE_ROWS_SQL} AS v({', '.join(types_columns)})",
        )
    )
    # narrow and unusual types; `fl` holds single-precision values, which a double
    # literal such as 1.1 is not equal to
    small_columns = {
        "id": "BIGINT",
        "ty": "TINYINT",
        "fl": "FLOAT",
        "iv": "INTERVAL DAY TO SECOND",
        "ym": "INTERVAL YEAR TO MONTH",
        "ch": "CHAR(5)",
        "vc": "VARCHAR(10)",
        "ast": "ARRAY<STRUCT<a: INT, b: STRING>>",
        "mpd": "MAP<STRING, DECIMAL(10,2)>",
    }
    run(
        f"CREATE OR REPLACE TABLE {t('edge_small')} AS "
        + _select(
            small_columns,
            f"""FROM VALUES
        (1, CAST(-128 AS TINYINT), CAST(1.1 AS FLOAT), INTERVAL '1 02:03:04.5' DAY TO SECOND,
            INTERVAL '1-2' YEAR TO MONTH, 'ab', 'cd',
            array(named_struct('a', 1, 'b', 'x'), named_struct('a', CAST(NULL AS INT), 'b', 'y')),
            map('k', CAST('1.50' AS DECIMAL(10,2)))),
        (2, CAST(127 AS TINYINT), CAST(0.1 AS FLOAT), INTERVAL '-1 00:00:00' DAY TO SECOND,
            INTERVAL '-0-1' YEAR TO MONTH, 'abcde', '',
            array(), map()),
        (3, CAST(0 AS TINYINT), CAST('16777217' AS FLOAT), INTERVAL '0 00:00:00' DAY TO SECOND,
            INTERVAL '0-0' YEAR TO MONTH, '', 'x ',
            NULL, NULL),
        (4, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
        AS v({", ".join(small_columns)})""",
        )
    )
    # America/Los_Angeles: 02:30 on 2024-03-10 doesn't exist (the clocks jump to 03:00);
    # 01:30 on 2024-11-03 happens twice. The `ltz` column holds the same instants.
    zones_columns = {
        "id": "BIGINT",
        "tag": "STRING",
        "ntz": "TIMESTAMP_NTZ",
        "ltz": "TIMESTAMP",
    }
    run(
        f"CREATE OR REPLACE TABLE {t('edge_zones')} AS "
        + _select(
            zones_columns,
            f"""FROM VALUES
        (1, 'before gap', TIMESTAMP_NTZ '2024-03-10 01:30:00',
            CAST('2024-03-10 09:30:00+00:00' AS TIMESTAMP)),
        (2, 'in gap', TIMESTAMP_NTZ '2024-03-10 02:30:00',
            CAST('2024-03-10 10:30:00+00:00' AS TIMESTAMP)),
        (3, 'after gap', TIMESTAMP_NTZ '2024-03-10 03:30:00',
            CAST('2024-03-10 10:30:00+00:00' AS TIMESTAMP)),
        (4, 'ambiguous', TIMESTAMP_NTZ '2024-11-03 01:30:00',
            CAST('2024-11-03 08:30:00+00:00' AS TIMESTAMP)),
        (5, 'summer', TIMESTAMP_NTZ '2024-07-01 12:00:00',
            CAST('2024-07-01 19:00:00+00:00' AS TIMESTAMP)),
        (6, 'winter', TIMESTAMP_NTZ '2024-01-01 12:00:00',
            CAST('2024-01-01 20:00:00+00:00' AS TIMESTAMP)),
        (7, 'null', NULL, NULL)
        AS v({", ".join(zones_columns)})""",
        )
    )
    collation_columns = {
        "id": "BIGINT",
        "s": "STRING",
        "lc": "STRING COLLATE UTF8_LCASE",
        "ci": "STRING COLLATE UNICODE_CI",
        "ai": "STRING COLLATE UNICODE_CI_AI",
        "u": "STRING COLLATE UNICODE",
    }
    values = ["abc", "ABC", "Abc", "abd", "é", "e", "É", "z", "B", "b", ""]
    rows = [(i, *[v] * 5) for i, v in enumerate(values)] + [(99, *[None] * 5)]
    run(
        f"CREATE OR REPLACE TABLE {t('edge_collations')} AS "
        + _select(collation_columns, _values(rows, list(collation_columns)))
    )
    # the string columns take the table's default collation (as the catalog shows them);
    # `b` is an explicit binary one
    default_rows = [
        (1, "abc", "abc"),
        (2, "ABC", "ABC"),
        (3, "x", "x"),
        (4, None, None),
    ]
    run(
        f"CREATE OR REPLACE TABLE {t('edge_default_collation')} "
        "DEFAULT COLLATION UTF8_LCASE AS "
        + _select(
            {"id": "BIGINT", "s": "", "b": "STRING COLLATE UTF8_BINARY"},
            _values(default_rows, ["id", "s", "b"]),
        )
    )
    chars = [(i, v) for i, v in enumerate(CHAR_VALUES)] + [(len(CHAR_VALUES), None)]
    run(
        f"CREATE OR REPLACE TABLE {t('edge_chars')} AS "
        + _select({"id": "BIGINT", "s": "STRING"}, _values(chars, ["id", "s"]))
    )
    run(
        f"CREATE OR REPLACE VIEW {t('rand_view')} AS "
        "SELECT id, rand() AS r FROM range(1, 101) AS t(id)"
    )
    run(
        f"CREATE OR REPLACE VIEW {t('plain_view')} AS "
        "SELECT id, id * 2 AS twice FROM range(1, 101) AS t(id)"
    )
    # a view over a view: its text names `plain_view` unqualified
    run(
        f"CREATE OR REPLACE VIEW {t('nested_plain_view')} AS "
        "SELECT id, twice FROM plain_view"
    )
    # last: a seeder that stopped halfway leaves no current version
    run(
        f"CREATE OR REPLACE TABLE {t('_seed_version')} AS SELECT '{VERSION}' AS version"
    )


def _seeded(conn: Any, catalog: str, schema: str) -> bool:
    name = f"{_ident(catalog)}.{_ident(schema)}._seed_version"
    try:
        return bool(execute(conn, f"SELECT version FROM {name}")[0][0] == VERSION)
    except Exception:
        return False


def _seed_if_needed(uri: str) -> None:
    catalog, schema = _target(uri)
    conn = connect(uri)
    try:
        if _seeded(conn, catalog, schema):
            return
        try:
            seed(conn, catalog, schema)
        except RuntimeError:
            raise
        except Exception:
            # a seeder in another process (a CI job, maybe of another version) replaced
            # a table at the same time (a concurrent Delta commit is refused): wait for
            # one to finish, up to 5 minutes, then seed again
            for _ in range(30):
                time.sleep(10)
                if _seeded(conn, catalog, schema):
                    return
            seed(conn, catalog, schema)
    finally:
        with contextlib.suppress(Exception):
            conn.close()


@functools.cache
def ensure_seeded(uri: str) -> None:
    """
    Seeds the schema unless it is current; once per process, one process at a time.

    The tests run on several xdist workers: the first takes a file lock, the others wait
    and find it seeded. (The lock is of one machine: seeders on several, and Windows,
    rely on every table being one statement.)
    """
    _target(uri)
    if fcntl is None:
        _seed_if_needed(uri)
        return
    lock = Path(tempfile.gettempdir()) / "polars-test-databricks-seed.lock"
    with lock.open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            _seed_if_needed(uri)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


if __name__ == "__main__":
    uri = os.environ.get(ENV)
    if not uri:
        sys.exit(f"{ENV} is not set")
    catalog, schema = _target(uri)
    c = connect(uri)
    seed(c, catalog, schema)
    c.close()
    print("seeded", catalog, schema)
