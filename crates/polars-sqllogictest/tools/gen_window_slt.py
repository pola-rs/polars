"""
Generate the window function tests in `slt/window/`.

The expected results come from DuckDB, set to the PostgreSQL NULL order that Polars SQL
uses. Each query is also run on Polars (through `polars-sqllogictest --queries`):

- queries that Polars rejects are left out, until a change makes them work;
- queries where Polars gives a different answer are written to the window section of
  `expected_failures.txt`.

Only queries with one correct answer are generated: functions that depend on the order
of rows get a total ORDER BY.

Run it again after a change that makes Polars accept more queries:

    python crates/polars-sqllogictest/tools/gen_window_slt.py

It needs `duckdb` and a Rust toolchain.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb

CRATE_DIR = Path(__file__).resolve().parent.parent
SLT_DIR = CRATE_DIR / "slt" / "window"
EXPECTED_FAILURES = CRATE_DIR / "expected_failures.txt"
BEGIN_MARKER = "# BEGIN window (written by tools/gen_window_slt.py)"
END_MARKER = "# END window"

ROWS = [
    # id, g, h, x, y, s
    (1, 1, "a", 5, 1.5, "p"),
    (2, 1, "b", 3, None, "q"),
    (3, 1, "a", 5, 2.0, None),
    (4, 2, "a", None, 3.25, "r"),
    (5, 2, "b", 2, -1.0, "p"),
    (6, 2, "b", 2, 2.0, "s"),
    (7, 2, "a", 8, 0.5, "q"),
    (8, 3, "a", 1, None, "t"),
    (9, 3, None, 1, 7.0, "u"),
    (10, None, "a", 4, 2.0, None),
    (11, None, "b", None, 1.0, "p"),
    (12, 1, "b", 3, 4.5, "v"),
    (13, 2, "a", 9, -3.0, "w"),
    (14, 3, "a", 1, 0.0, "q"),
]
U_ROWS = [(1, "one"), (2, "two"), (3, "three"), (2, "two-bis")]


def sql_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return f"'{v}'"
    return repr(v)


def values(rows: list[tuple[Any, ...]]) -> str:
    return ", ".join("(" + ", ".join(sql_literal(v) for v in r) + ")" for r in rows)


SETUP = [
    "CREATE TABLE t (id BIGINT, g BIGINT, h VARCHAR, x BIGINT, y DOUBLE, s VARCHAR)",
    f"INSERT INTO t VALUES {values(ROWS)}",
    "CREATE TABLE e (id BIGINT, g BIGINT, h VARCHAR, x BIGINT, y DOUBLE, s VARCHAR)",
    "CREATE TABLE u (g BIGINT, label VARCHAR)",
    f"INSERT INTO u VALUES {values(U_ROWS)}",
]

# (window, has ORDER BY, ORDER BY is total, single numeric ORDER BY key)
SPECS = [
    ("", False, False, False),
    ("PARTITION BY g", False, False, False),
    ("PARTITION BY g, h", False, False, False),
    ("PARTITION BY h", False, False, False),
    ("PARTITION BY g % 2", False, False, False),
    ("ORDER BY x", True, False, True),
    ("ORDER BY x DESC", True, False, True),
    ("ORDER BY x NULLS FIRST", True, False, True),
    ("ORDER BY x DESC NULLS LAST", True, False, True),
    ("ORDER BY s", True, False, False),
    ("ORDER BY y DESC", True, False, True),
    ("ORDER BY h, x DESC", True, False, False),
    ("ORDER BY id", True, True, True),
    ("ORDER BY x, id", True, True, False),
    ("ORDER BY x DESC, id DESC", True, True, False),
    ("ORDER BY -x, id", True, True, False),
    ("PARTITION BY g ORDER BY x", True, False, True),
    ("PARTITION BY g ORDER BY x DESC NULLS FIRST", True, False, True),
    ("PARTITION BY g ORDER BY x, id", True, True, False),
    ("PARTITION BY g ORDER BY x DESC, id", True, True, False),
    ("PARTITION BY g ORDER BY y DESC NULLS LAST, id", True, True, False),
    ("PARTITION BY h ORDER BY s, id", True, True, False),
    ("PARTITION BY g, h ORDER BY x", True, False, True),
    ("PARTITION BY h ORDER BY id", True, True, True),
]

# (frame, needs a total ORDER BY, kind)
FRAMES = [
    ("ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", True, "rows"),
    ("ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING", True, "rows"),
    ("ROWS BETWEEN 2 PRECEDING AND CURRENT ROW", True, "rows"),
    ("ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING", True, "rows"),
    ("ROWS BETWEEN 1 FOLLOWING AND 3 FOLLOWING", True, "rows"),
    ("ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING", False, "rows"),
    ("ROWS 2 PRECEDING", True, "rows"),
    ("RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW", False, "range"),
    ("RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING", False, "range"),
    ("RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING", False, "range"),
    ("RANGE BETWEEN 2 PRECEDING AND 2 FOLLOWING", False, "range_offset"),
    ("RANGE BETWEEN CURRENT ROW AND CURRENT ROW", False, "range"),
    ("GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW", False, "groups"),
]

RANKING = [
    "ROW_NUMBER()",
    "RANK()",
    "DENSE_RANK()",
    "PERCENT_RANK()",
    "CUME_DIST()",
    "NTILE(3)",
]
DEPENDS_ON_ROW_ORDER = {"ROW_NUMBER()", "NTILE(3)"}
OFFSET = [
    "LAG(x)",
    "LAG(x, 2)",
    "LAG(x, 1, -1)",
    "LEAD(x)",
    "LEAD(y, 2, 0.5)",
    "LAG(s)",
]
VALUE = ["FIRST_VALUE(x)", "LAST_VALUE(x)", "NTH_VALUE(x, 2)", "FIRST_VALUE(s)"]
AGGS = ["SUM(x)", "SUM(y)", "AVG(x)", "MIN(y)", "MAX(s)", "COUNT(*)", "COUNT(y)"]

# Window functions inside larger queries: (file, sql, result order matters)
QUERIES = [
    (
        "over_group_by",
        "SELECT g, SUM(x) AS sx, SUM(SUM(x)) OVER (ORDER BY g) AS w FROM t GROUP BY g",
    ),
    (
        "over_group_by",
        "SELECT g, COUNT(*) AS c, RANK() OVER (ORDER BY COUNT(*) DESC) AS w FROM t GROUP BY g",
    ),
    (
        "over_group_by",
        "SELECT g, SUM(x) AS sx, ROW_NUMBER() OVER (ORDER BY SUM(x) DESC NULLS LAST, g) AS w FROM t GROUP BY g HAVING COUNT(*) > 1",
    ),
    (
        "over_group_by",
        "SELECT h, AVG(y) AS a, AVG(AVG(y)) OVER () AS w FROM t GROUP BY h",
    ),
    (
        "over_group_by",
        "SELECT g, h, COUNT(*) AS c, SUM(COUNT(*)) OVER (PARTITION BY g) AS w FROM t GROUP BY g, h",
    ),
    ("expressions", "SELECT id, x - AVG(x) OVER (PARTITION BY g) AS w FROM t"),
    (
        "expressions",
        "SELECT id, CASE WHEN ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) = 1 THEN 'first' ELSE 'other' END AS w FROM t",
    ),
    ("expressions", "SELECT id, 100.0 * x / SUM(x) OVER (PARTITION BY g) AS w FROM t"),
    (
        "expressions",
        "SELECT id, COALESCE(LAG(x) OVER (ORDER BY id), 0) + x AS w FROM t",
    ),
    ("expressions", "SELECT id, SUM(x * 2 + 1) OVER (PARTITION BY g) AS w FROM t"),
    (
        "expressions",
        "SELECT id, SUM(CASE WHEN x > 2 THEN 1 ELSE 0 END) OVER (PARTITION BY g ORDER BY id) AS w FROM t",
    ),
    (
        "expressions",
        "SELECT id, RANK() OVER (ORDER BY x) + DENSE_RANK() OVER (ORDER BY y) AS w FROM t",
    ),
    (
        "expressions",
        "SELECT id, SUM(x) OVER (PARTITION BY g + 1 ORDER BY x * -1, id) AS w FROM t",
    ),
    (
        "expressions",
        "SELECT id, RANK() OVER (PARTITION BY COALESCE(h, 'z') ORDER BY ABS(y) DESC NULLS LAST) AS w FROM t",
    ),
    ("expressions", "SELECT id, COUNT(*) OVER (PARTITION BY x IS NULL) AS w FROM t"),
    (
        "qualify",
        "SELECT id FROM t QUALIFY ROW_NUMBER() OVER (PARTITION BY g ORDER BY x DESC, id) = 1",
    ),
    (
        "qualify",
        "SELECT id, x FROM t QUALIFY RANK() OVER (ORDER BY x DESC NULLS LAST) <= 3",
    ),
    (
        "qualify",
        "SELECT * FROM (SELECT id, RANK() OVER (ORDER BY x) AS r FROM t) AS q WHERE r <= 3",
    ),
    ("qualify", "SELECT id FROM t QUALIFY SUM(x) OVER (PARTITION BY g) > 10"),
    (
        "named_windows",
        "SELECT id, SUM(x) OVER w AS a, ROW_NUMBER() OVER w AS b FROM t WINDOW w AS (PARTITION BY g ORDER BY id)",
    ),
    (
        "named_windows",
        "SELECT id, SUM(x) OVER (w ORDER BY id) AS a FROM t WINDOW w AS (PARTITION BY g)",
    ),
    (
        "named_windows",
        "SELECT id, COUNT(*) OVER w AS a, SUM(x) OVER w AS b FROM t WINDOW w AS (PARTITION BY g ORDER BY x)",
    ),
    ("named_windows", "SELECT id, AVG(y) OVER w AS a FROM t WINDOW w AS (ORDER BY id)"),
    (
        "named_windows",
        "SELECT id, RANK() OVER (w ORDER BY x DESC) AS a FROM t WINDOW w AS (PARTITION BY h)",
    ),
    (
        "queries",
        "SELECT id, SUM(x) OVER (PARTITION BY g) AS a, SUM(x) OVER (PARTITION BY h) AS b, COUNT(*) OVER () AS c FROM t",
    ),
    (
        "queries",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS a, ROW_NUMBER() OVER (ORDER BY id DESC) AS b FROM t",
    ),
    (
        "queries",
        "SELECT id, MIN(x) OVER (PARTITION BY g ORDER BY id) AS a, MAX(x) OVER (PARTITION BY h ORDER BY id DESC) AS b FROM t",
    ),
    ("queries", "SELECT id, SUM(x) OVER (PARTITION BY g) AS w FROM t WHERE y > 0"),
    (
        "queries",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS w FROM t WHERE x IS NOT NULL",
    ),
    (
        "queries",
        "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t ORDER BY rn LIMIT 5",
        True,
    ),
    (
        "queries",
        "SELECT id, SUM(x) OVER (ORDER BY id) AS w FROM t ORDER BY w DESC NULLS LAST, id LIMIT 4",
        True,
    ),
    (
        "queries",
        "SELECT id FROM t ORDER BY ROW_NUMBER() OVER (ORDER BY x DESC NULLS LAST, id)",
        True,
    ),
    ("queries", "SELECT DISTINCT g, SUM(x) OVER (PARTITION BY g) AS w FROM t"),
    ("queries", "SELECT DISTINCT h, COUNT(*) OVER (PARTITION BY h) AS w FROM t"),
    (
        "queries",
        "SELECT t.id, u.label, COUNT(*) OVER (PARTITION BY u.label) AS w FROM t JOIN u ON t.g = u.g",
    ),
    (
        "queries",
        "SELECT t.id, u.label, ROW_NUMBER() OVER (PARTITION BY t.id ORDER BY u.label) AS w FROM t JOIN u ON t.g = u.g",
    ),
    (
        "queries",
        "SELECT t.id, SUM(t.x) OVER (PARTITION BY u.g) AS w FROM t LEFT JOIN u ON t.g = u.g",
    ),
    ("queries", "SELECT id, SUM(x) OVER () AS w FROM t WHERE 1 = 0"),
    ("queries", "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS w FROM e"),
    ("queries", "SELECT id, LAG(x) OVER (PARTITION BY g ORDER BY id) AS w FROM e"),
    (
        "queries",
        "SELECT id, SUM(r) OVER (PARTITION BY g) AS w FROM (SELECT id, g, RANK() OVER (ORDER BY x) AS r FROM t) AS q",
    ),
    (
        "queries",
        "WITH c AS (SELECT id, g, x - LAG(x) OVER (PARTITION BY g ORDER BY id) AS d FROM t) SELECT id, AVG(d) OVER (PARTITION BY g) AS w FROM c",
    ),
    (
        "queries",
        "SELECT g, MAX(rn) AS w FROM (SELECT g, ROW_NUMBER() OVER (PARTITION BY g ORDER BY id) AS rn FROM t) AS q GROUP BY g",
    ),
    ("arguments", "SELECT id, NTILE(20) OVER (ORDER BY id) AS w FROM t"),
    ("arguments", "SELECT id, NTILE(1) OVER (PARTITION BY g ORDER BY id) AS w FROM t"),
    ("arguments", "SELECT id, LAG(x, 0) OVER (ORDER BY id) AS w FROM t"),
    ("arguments", "SELECT id, LEAD(x, 20) OVER (ORDER BY id) AS w FROM t"),
    ("arguments", "SELECT id, LAG(y, 1, 0) OVER (ORDER BY id) AS w FROM t"),
    ("arguments", "SELECT id, LAG(x, 1, x) OVER (ORDER BY id) AS w FROM t"),
    (
        "arguments",
        "SELECT id, NTH_VALUE(x, 3) OVER (PARTITION BY g ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS w FROM t",
    ),
    ("arguments", "SELECT id, NTH_VALUE(x, 10) OVER (ORDER BY id) AS w FROM t"),
    (
        "arguments",
        "SELECT id, PERCENT_RANK() OVER (PARTITION BY id ORDER BY x) AS w FROM t",
    ),
    (
        "arguments",
        "SELECT id, CUME_DIST() OVER (PARTITION BY g ORDER BY x DESC) AS w FROM t",
    ),
    ("arguments", "SELECT id, LAG(x) IGNORE NULLS OVER (ORDER BY id) AS w FROM t"),
    (
        "arguments",
        "SELECT id, FIRST_VALUE(y) IGNORE NULLS OVER (PARTITION BY g ORDER BY id) AS w FROM t",
    ),
    (
        "arguments",
        "SELECT id, LAST_VALUE(y IGNORE NULLS) OVER (ORDER BY id) AS w FROM t",
    ),
    ("aggregates", "SELECT id, COUNT(DISTINCT x) OVER (PARTITION BY g) AS w FROM t"),
    ("aggregates", "SELECT id, SUM(DISTINCT x) OVER (PARTITION BY g) AS w FROM t"),
    (
        "aggregates",
        "SELECT id, COUNT(*) FILTER (WHERE x > 2) OVER (PARTITION BY g) AS w FROM t",
    ),
    (
        "aggregates",
        "SELECT id, SUM(x) FILTER (WHERE y > 0) OVER (ORDER BY id) AS w FROM t",
    ),
    ("aggregates", "SELECT id, STDDEV(y) OVER (PARTITION BY g) AS w FROM t"),
    (
        "aggregates",
        "SELECT id, VARIANCE(y) OVER (PARTITION BY h ORDER BY id) AS w FROM t",
    ),
    ("aggregates", "SELECT id, MEDIAN(y) OVER (PARTITION BY g) AS w FROM t"),
    ("aggregates", "SELECT id, BOOL_AND(x > 2) OVER (PARTITION BY g) AS w FROM t"),
    (
        "aggregates",
        "SELECT id, STRING_AGG(s, ',' ORDER BY s) OVER (PARTITION BY g) AS w FROM t",
    ),
    ("aggregates", "SELECT id, MIN(s) OVER (ORDER BY s NULLS FIRST) AS w FROM t"),
    (
        "frames",
        "SELECT id, MAX(id) OVER (PARTITION BY g ORDER BY y RANGE BETWEEN 1.5 PRECEDING AND 1.5 FOLLOWING) AS w FROM t",
    ),
]


def generate() -> list[dict[str, Any]]:
    queries = []

    def add(file: str, sql: str, ordered: bool = False) -> None:
        queries.append({"file": file, "sql": sql, "ordered": ordered})

    for spec, has_order, total, _ in SPECS:
        over = f"OVER ({spec})"
        for f in RANKING:
            if has_order and (total or f not in DEPENDS_ON_ROW_ORDER):
                add("ranking", f"SELECT id, {f} {over} AS w FROM t")
        for f in OFFSET:
            if total:
                add("lag_lead", f"SELECT id, {f} {over} AS w FROM t")
        for f in VALUE:
            if total:
                add("value_functions", f"SELECT id, {f} {over} AS w FROM t")
        for f in AGGS:
            add("aggregates", f"SELECT id, {f} {over} AS w FROM t")

    for spec, has_order, total, single_numeric_key in SPECS:
        if not has_order:
            continue
        for frame, needs_total, kind in FRAMES:
            if kind == "range_offset" and not single_numeric_key:
                continue
            over = f"OVER ({spec} {frame})"
            if total or not needs_total:
                for f in AGGS:
                    add("frames", f"SELECT id, {f} {over} AS w FROM t")
            if total:
                for f in VALUE:
                    add("frames", f"SELECT id, {f} {over} AS w FROM t")

    for file, sql, *ordered in QUERIES:
        add(file, sql, bool(ordered and ordered[0]))
    return queries


def column_kind(duckdb_type: str) -> str:
    """Column type letter of a `query` record: I(nteger), R(eal) or T(ext)."""
    if duckdb_type == "BOOLEAN" or "INT" in duckdb_type:
        return "I"
    if duckdb_type in ("DOUBLE", "FLOAT", "REAL") or duckdb_type.startswith("DECIMAL"):
        return "R"
    return "T"


def format_value(v: Any, kind: str) -> str:
    """Format a value the way the harness formats Polars results (see `src/output.rs`)."""
    if v is None:
        return "NULL"
    if kind == "R":
        v = float(v)
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return f"{v:.3f}"
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, Decimal)):
        return str(v)
    s = str(v)
    return s if s else "(empty)"


def run_duckdb(queries: list[dict[str, Any]]) -> None:
    con = duckdb.connect()
    con.execute("SET default_null_order = 'nulls_last_on_asc_first_on_desc'")
    for sql in SETUP:
        con.execute(sql)
    for q in queries:
        try:
            cur = con.execute(q["sql"])
            rows = cur.fetchall()
        except duckdb.Error:
            q["expected"] = None
            continue
        kinds = [column_kind(str(d[1])) for d in cur.description]
        q["kinds"] = "".join(kinds)
        q["expected"] = [
            [format_value(v, k) for v, k in zip(row, kinds, strict=True)]
            for row in rows
        ]


def run_polars(queries: list[dict[str, Any]]) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "queries.sql"
        path.write_text("\n;;\n".join(SETUP + [q["sql"] for q in queries]) + "\n")
        out = subprocess.run(
            [
                "cargo",
                "run",
                "--quiet",
                "-p",
                "polars-sqllogictest",
                "--",
                "--queries",
                str(path),
            ],
            cwd=CRATE_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    results = [json.loads(line) for line in out.splitlines()]
    assert len(results) == len(SETUP) + len(queries), out[-2000:]
    for sql, res in zip(SETUP, results, strict=False):
        assert res["ok"], (sql, res)
    for q, res in zip(queries, results[len(SETUP) :], strict=True):
        q["actual"] = res["rows"] if res["ok"] else None


def has_whitespace(rows: list[list[str]]) -> bool:
    return any(any(c.isspace() for c in v) for row in rows for v in row)


def write_files(queries: list[dict[str, Any]]) -> list[str]:
    """Write the .slt files; returns the keys of the queries Polars gets wrong."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for q in queries:
        by_file.setdefault(q["file"], []).append(q)

    for old in SLT_DIR.glob("*.slt"):
        old.unlink()
    SLT_DIR.mkdir(parents=True, exist_ok=True)

    wrong = []
    for file, file_queries in sorted(by_file.items()):
        lines = ["# Generated by tools/gen_window_slt.py; do not edit.", ""]
        for sql in SETUP:
            lines += ["statement ok", sql, ""]
        for q in file_queries:
            expected = q["expected"]
            actual = q["actual"]
            if not q["ordered"]:
                expected = sorted(expected)
                actual = sorted(actual)
            if actual != expected:
                wrong.append(f"window/{file}.slt:{len(lines) + 1}")
            sort_mode = "" if q["ordered"] else " rowsort"
            lines += [f"query {q['kinds']}{sort_mode}", q["sql"], "----"]
            lines += ["\t".join(row) for row in expected]
            lines.append("")
        (SLT_DIR / f"{file}.slt").write_text("\n".join(lines))
    return wrong


def write_expected_failures(wrong: list[str]) -> None:
    text = EXPECTED_FAILURES.read_text()
    if BEGIN_MARKER in text:
        before, rest = text.split(BEGIN_MARKER, 1)
        after = rest.split(END_MARKER, 1)[1]
    else:
        before, after = text.rstrip("\n") + "\n\n", "\n"
    block = "\n".join(
        [
            BEGIN_MARKER,
            "# Window functions that give a wrong result.",
            *wrong,
            END_MARKER,
        ]
    )
    EXPECTED_FAILURES.write_text(before + block + after)


def main() -> None:
    queries = generate()
    run_duckdb(queries)
    queries = [q for q in queries if q["expected"] is not None]
    run_polars(queries)
    rejected = [q for q in queries if q["actual"] is None]
    queries = [q for q in queries if q["actual"] is not None]
    for q in queries:
        assert not has_whitespace(q["expected"]), q["sql"]

    wrong = write_files(queries)
    write_expected_failures(wrong)
    print(
        f"{len(queries)} queries written ({len(wrong)} wrong in Polars), "
        f"{len(rejected)} left out because Polars rejects them"
    )


if __name__ == "__main__":
    main()
