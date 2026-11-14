"""
The sqlglot SQL renderer behind the Rust `SqlRenderer` trait.

Rust sends a dialect-neutral `SqlQuery` as JSON; this module builds sqlglot
expressions from it and generates SQL for the target dialect. It is the only code
that imports sqlglot, and it only runs on the machine that plans the query.
"""

from __future__ import annotations

import contextlib
import json
import secrets
from typing import Any

from polars._dependencies import import_optional

# The lowest version the tests pass with (also in pyproject.toml): before 28.6, Oracle's
# SYSTIMESTAMP isn't parsed as a volatile function, so a query on it was read as
# repeatable.
sqlglot = import_optional(
    "sqlglot",
    err_prefix="scan_database needs",
    min_version="28.6",
    install_message="Install it with: pip install -U sqlglot",
)
exp = sqlglot.exp

_BINARY = {
    "Eq": exp.EQ,
    "NotEq": exp.NEQ,
    "Lt": exp.LT,
    "LtEq": exp.LTE,
    "Gt": exp.GT,
    "GtEq": exp.GTE,
    "And": exp.And,
    "Or": exp.Or,
}

# Temporal literals are cast with microsecond precision, so that no dialect rounds them.
_TIMESTAMP = "TIMESTAMP(6)"
_TIMESTAMPTZ = "TIMESTAMPTZ(6)"
_TIME = "TIME(6)"
# Spark SQL's timestamps are always microseconds and take no precision. sqlglot drops it
# for these dialects up to 30.24, but keeps it for Databricks from 30.25.
_NO_TEMPORAL_PRECISION = {"databricks", "spark", "hive"}


def version() -> str:
    return str(sqlglot.__version__)


def _ident(name: str) -> Any:
    return exp.to_identifier(name, quoted=True)


# ClickHouse reads `\` as an escape in a quoted identifier too. sqlglot escapes it there
# only from 30.22: before, a column `x\` was written `"x\"`, which reads as an escaped
# quote.
_CLICKHOUSE_ESCAPES_BACKSLASH = (
    exp.to_identifier("\\", quoted=True).sql("clickhouse") == '"\\\\"'
)


def _escaped_identifiers(node: Any, dialect: str | None) -> Any:
    """`node`, its ClickHouse identifiers' backslashes escaped where sqlglot doesn't."""
    if dialect == "clickhouse" and not _CLICKHOUSE_ESCAPES_BACKSLASH:
        for ident in node.find_all(exp.Identifier):
            if ident.quoted and "\\" in ident.this:
                ident.set("this", ident.this.replace("\\", "\\\\"))
    return node


def _column(name: str) -> Any:
    return exp.Column(this=_ident(name))


def _raw(sql: str) -> Any:
    # inserted verbatim: user-written SQL is never reparsed or rewritten
    return exp.Var(this=sql)


def _cast(text: str, dtype: str) -> Any:
    return exp.Cast(this=exp.Literal.string(text), to=exp.DataType.build(dtype))


def _ansi(kind: str, text: str) -> Any:
    # `DATE '…'` / `TIMESTAMP '…'`: Oracle's CAST from text depends on NLS settings
    return _raw(f"{kind} '{text}'")


def _string(text: str, dialect: str | None) -> Any:
    if dialect == "tsql":
        # N'…': a plain literal is converted to the database's code page first, which
        # can turn characters it lacks into '?'
        return exp.National(this=text)
    if dialect == "postgres" and "\\" in text:
        # E'…' reads backslashes the same whatever `standard_conforming_strings` is; a
        # plain '…' holding one means something else when it is off
        escaped = text.replace("\\", "\\\\").replace("'", "''")
        return _raw(f"E'{escaped}'")
    return exp.Literal.string(text)


def _number(text: str) -> Any:
    # the text as written: `exp.Literal.number` makes a negative number a `Neg` node,
    # which sqlglot writes through a 28-digit `Decimal` (rounding wider ones)
    return exp.Literal(this=text, is_string=False)


def _literal(v: Any, dialect: str | None) -> Any:
    if v == "Null":
        return exp.Null()
    ((kind, x),) = v.items()
    if kind == "Bool":
        return exp.Boolean(this=x)
    if kind == "Int":
        return _number(str(int(x)))
    if kind == "Float":
        return _number(repr(float(x)))
    if kind == "Decimal":
        if dialect == "clickhouse":
            # a bare literal is parsed as a float, which overflows for wide decimals
            scale = len(x.partition(".")[2])
            return _cast(x, f"Decimal(38, {scale})")
        if dialect == "bigquery":
            # a bare `1.5` is a FLOAT64, which a NUMERIC column would be compared as;
            # NUMERIC holds 29 integer and 9 fractional digits, BIGNUMERIC 38 of each
            whole, _, frac = x.lstrip("-").partition(".")
            fits = len(frac) <= 9 and len(whole.lstrip("0")) <= 29
            return _raw(f"{'NUMERIC' if fits else 'BIGNUMERIC'} '{x}'")
        if dialect == "mysql" and "." in x:
            # MySQL 8 folds a bare literal finer than a DECIMAL column's scale to that
            # scale with the wrong sign (`x < -0.005` keeps 0.00); a typed one compares
            # exactly. DECIMAL holds 65 digits, 30 of them fractional.
            whole, _, frac = x.lstrip("-").partition(".")
            frac = frac.rstrip("0")
            if len(frac) <= 30 and len(whole.lstrip("0")) + len(frac) <= 65:
                return _cast(x, f"DECIMAL(65, {len(frac)})")
        return _number(x)
    if kind == "String":
        return _string(x, dialect)
    if kind == "Date":
        if dialect == "oracle":
            return _ansi("DATE", x)
        if dialect == "clickhouse":
            # `Date` starts in 1970 and wraps silently; `Date32` covers 1900-2299
            return exp.Cast(
                this=exp.Literal.string(x),
                to=exp.DataType.build("Date32", dialect="clickhouse"),
            )
        return _cast(x, "DATE")
    if kind == "Timestamp":
        if dialect == "oracle":
            return _ansi("TIMESTAMP", x["text"] + (" +00:00" if x["with_tz"] else ""))
        if dialect == "clickhouse":
            return exp.Cast(
                this=exp.Literal.string(x["text"]),
                to=exp.DataType.build(
                    "DateTime64(6, 'UTC')" if x["with_tz"] else "DateTime64(6)",
                    dialect="clickhouse",
                ),
            )
        if dialect in _NO_TEMPORAL_PRECISION:
            text = x["text"] + ("+00:00" if x["with_tz"] else "")
            return _cast(text, "TIMESTAMP")
        if x["with_tz"]:
            return _cast(x["text"] + "+00:00", _TIMESTAMPTZ)
        return _cast(x["text"], _TIMESTAMP)
    if kind == "Time":
        return _cast(x, "TIME" if dialect in _NO_TEMPORAL_PRECISION else _TIME)
    msg = f"unknown SQL literal kind: {kind!r}"
    raise ValueError(msg)


def _expr(e: Any, d: str | None) -> Any:
    ((kind, x),) = e.items()
    if kind == "Column":
        return _column(x)
    if kind == "Literal":
        return _literal(x, d)
    if kind == "Binary":
        left, op, right = x
        node = _BINARY[op]
        lhs, rhs = _expr(left, d), _expr(right, d)
        if op in ("And", "Or"):
            lhs, rhs = exp.Paren(this=lhs), exp.Paren(this=rhs)
        return node(this=lhs, expression=rhs)
    if kind == "Not":
        return exp.Not(this=exp.Paren(this=_expr(x, d)))
    if kind == "IsNull":
        inner, negated = x
        is_null = exp.Is(this=_expr(inner, d), expression=exp.Null())
        # `NOT (…)`: MySQL's `HIGH_NOT_PRECEDENCE` reads `NOT x IS NULL` as `(NOT x) IS
        # NULL`
        return exp.Not(this=exp.Paren(this=is_null)) if negated else is_null
    if kind == "InList":
        inner, values, negated = x
        node = exp.In(
            this=_expr(inner, d), expressions=[_literal(v, d) for v in values]
        )
        return exp.Not(this=exp.Paren(this=node)) if negated else node
    if kind == "Between":
        inner, lo, hi = x
        return exp.Between(this=_expr(inner, d), low=_expr(lo, d), high=_expr(hi, d))
    if kind == "Like":
        pattern = _string(x["pattern"], d)
        like = exp.Like(this=_expr(x["expr"], d), expression=pattern)
        if d == "clickhouse" and x["escape"] == "\\":
            # no ESCAPE clause; backslash is its escape character already
            return like
        escape = (
            exp.Literal.string(x["escape"]) if d == "tsql" else _string(x["escape"], d)
        )
        return exp.Escape(this=like, expression=escape)
    if kind == "IsNotDistinctFrom":
        left, right, not_distinct = x
        node = exp.NullSafeEQ if not_distinct else exp.NullSafeNEQ
        return node(this=_expr(left, d), expression=_expr(right, d))
    if kind == "Raw":
        return exp.Paren(this=_raw(x))
    msg = f"unknown SQL expression kind: {kind!r}"
    raise ValueError(msg)


def _select_item(item: Any) -> Any:
    if item == "Star":
        return exp.Star()
    if item == "CountStar":
        return exp.Count(this=exp.Star())
    if item == "One":
        return exp.Literal.number(1)
    ((kind, x),) = item.items()
    if kind == "Column":
        return _column(x)
    if kind == "Min":
        return exp.Min(this=_column(x))
    if kind == "Max":
        return exp.Max(this=_column(x))
    if kind == "Expr":
        # dialect conversion SQL from Polars' rules, e.g. CAST("u" AS TEXT)
        return exp.Alias(this=_raw(x["sql"]), alias=_ident(x["alias"]))
    msg = f"unknown SQL select item: {item!r}"
    raise ValueError(msg)


def _from(f: Any) -> Any:
    ((kind, x),) = f.items()
    if kind == "Table":
        parts = [_ident(p) for p in x]
        name = parts[-1]
        db = parts[-2] if len(parts) >= 2 else None
        catalog = parts[-3] if len(parts) >= 3 else None
        return exp.Table(this=name, db=db, catalog=catalog)
    if kind == "Subquery":
        # a placeholder, replaced by the verbatim user query after generation
        return exp.Table(
            this=exp.to_identifier(_SUBQUERY_PLACEHOLDER),
            alias=exp.TableAlias(this=_ident("_pl")),
        )
    msg = f"unknown SQL FROM kind: {kind!r}"
    raise ValueError(msg)


_SUBQUERY_PLACEHOLDER = f"__polars_subquery_{secrets.token_hex(8)}__"


def _qualified(node: Any, columns: set[str], from_: Any) -> Any:
    """
    `node` with its references to `columns` qualified by the relation: a select item
    reads them through a conversion under the same name, which `ORDER BY` (and
    ClickHouse's `WHERE`) would resolve the name to.
    """
    if not columns:
        return node
    ((kind, x),) = from_.items()
    if kind == "Table":
        parts = [_ident(p) for p in x]
        qualifier = {
            "table": parts[-1],
            "db": parts[-2] if len(parts) >= 2 else None,
            "catalog": parts[-3] if len(parts) >= 3 else None,
        }
    else:
        qualifier = {"table": _ident("_pl")}
    for column in list(node.find_all(exp.Column)):
        if not column.table and column.name in columns:
            column.set("table", qualifier["table"].copy())
            for k in ("db", "catalog"):
                if qualifier.get(k) is not None:
                    column.set(k, qualifier[k].copy())
    return node


def _ordered(o: Any, d: str | None, qualify: Any = lambda e: e) -> list[Any]:
    # NULLs where the dialect puts them by default, as range partitions expect
    # (`nulls_sort_first` in `partition.rs`)
    null_ordering = sqlglot.Dialect.get_or_raise(d).NULL_ORDERING
    desc = bool(o["descending"])
    nulls_first = (null_ordering == "nulls_are_small" and not desc) or (
        null_ordering == "nulls_are_large" and desc
    )
    this = qualify(_expr(o["expr"], d))
    if d is None:
        # an unknown database (FlightSQL, ODBC): its NULL order is unknown, and it may
        # not have `NULLS FIRST`. NULLs first on ASC, as range partitions expect
        is_null = exp.Case(
            ifs=[
                exp.If(
                    this=exp.Is(this=this.copy(), expression=exp.Null()),
                    true=exp.Literal.number(0),
                )
            ],
            default=exp.Literal.number(1),
        )
        return [
            exp.Ordered(this=is_null, desc=desc, nulls_first=nulls_first),
            exp.Ordered(this=this, desc=desc, nulls_first=nulls_first),
        ]
    ordered = exp.Ordered(this=this, desc=desc, nulls_first=nulls_first)
    if d in _NO_NULLS_CLAUSE or (
        sqlglot.Dialect.get_or_raise(d).generator_class.NULL_ORDERING_SUPPORTED is None
    ):
        # MySQL, SQL Server: no `NULLS FIRST`, and a NULL order no setting changes
        return [ordered]
    # Written even where it is the default, which sqlglot leaves out: a session setting
    # may change the default (DuckDB's `default_null_order`, Snowflake's
    # `DEFAULT_NULL_ORDERING`)
    sql = _escaped_identifiers(ordered, d).sql(dialect=d)
    if not sql.endswith(("NULLS FIRST", "NULLS LAST")):
        sql += " NULLS FIRST" if nulls_first else " NULLS LAST"
    return [_raw(sql)]


# Dialects that have `NULLS FIRST` but get none: SQLite only since 3.30, and its NULL
# order can't be changed
_NO_NULLS_CLAUSE = {"sqlite"}


def build(query: dict[str, Any], dialect: str | None = None) -> Any:
    """Build the sqlglot expression of a `SqlQuery`."""
    d = dialect
    columns = set(query.get("qualified") or ())

    def qualify(e: Any) -> Any:
        return _qualified(e, columns, query["from"])

    select = exp.Select(expressions=[_select_item(c) for c in query["columns"]])
    select = select.from_(_from(query["from"]), copy=False)
    if filters := query["filter"]:
        select = select.where(*(qualify(_expr(f, d)) for f in filters), copy=False)
    if order_by := query["order_by"]:
        select = select.order_by(
            *(e for o in order_by for e in _ordered(o, d, qualify)), copy=False
        )
    if (limit := query["limit"]) is not None:
        select = select.limit(int(limit), copy=False)
    return select


def render(query_json: str, dialect: str | None) -> str:
    """Render a `SqlQuery` (as JSON) for a dialect (`None` is sqlglot's default)."""
    query = json.loads(query_json)
    sql = str(_escaped_identifiers(build(query, dialect), dialect).sql(dialect=dialect))
    from_ = query["from"]
    if "Subquery" in from_:
        sub = from_["Subquery"]
        # a newline ends a trailing `--` (or MySQL `#`) comment of the user's query
        end = "\n" if "--" in sub or "#" in sub else ""
        sql = sql.replace(_SUBQUERY_PLACEHOLDER, f"({sub}{end})", 1)
    return sql


_WRITES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Merge,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Grant,
    exp.Command,
)


def _is_read_only(stmt: Any) -> bool:
    """A query that only reads: no DML or DDL (in CTEs either), no SELECT INTO."""
    if not isinstance(stmt, (exp.Query, exp.Values)):
        return False
    if stmt.find(*_WRITES) is not None:
        return False
    return not any(s.args.get("into") for s in stmt.find_all(exp.Select))


# Functions that return other values each time they run (or each time a query runs), by
# lower-cased name; sqlglot parses most of them as `Anonymous`. `random()` / `rand()` /
# `DBMS_RANDOM.VALUE` and the `uuid()` family parse as typed nodes (`_VOLATILE_NODES`).
_VOLATILE_FUNCTIONS = frozenset(
    {
        # Postgres (and DuckDB)
        "random",
        "random_normal",
        "setseed",
        "gen_random_uuid",
        "gen_random_bytes",
        "uuid_generate_v1",
        "uuid_generate_v1mc",
        "uuid_generate_v4",
        "uuidv4",
        "uuidv7",
        "nextval",
        # the session's (STABLE, but each partition reads on a session of its own)
        "pg_backend_pid",
        "inet_client_port",
        "inet_client_addr",
        "txid_current",
        "pg_current_xact_id",
        # the time of the call, statement or transaction: partitions, and a count and
        # its read, run in transactions of their own (typed nodes: `_VOLATILE_NODES`)
        "clock_timestamp",
        "statement_timestamp",
        "transaction_timestamp",
        "timeofday",
        "now",
        "current_timestamp",
        "get_current_timestamp",
        "today",
        # SQLite
        "randomblob",
        # SQL Server
        "newid",
        "newsequentialid",
        "crypt_gen_random",
        # Oracle (`DBMS_RANDOM.*` and `seq.NEXTVAL` are checked apart)
        "sys_guid",
        # MySQL / MariaDB
        "rand",
        "uuid",
        "uuid_short",
        "uuid_v4",
        "uuid_v7",
        "random_bytes",
        # the session's: each partition reads on a session of its own
        "connection_id",
        "last_insert_id",
        "sysdate",
        "curdate",
        "curtime",
        "unix_timestamp",
        "utc_timestamp",
        "localtimestamp",
        # SQL Server
        "getdate",
        "getutcdate",
        "sysdatetime",
        "sysutcdatetime",
        "sysdatetimeoffset",
        # ClickHouse (and its `rand*` family, see `_is_volatile`)
        "generateuuidv4",
        "generateuuidv7",
        "generateulid",
        "generateserialid",
        "generatesnowflakeid",
        "fuzzbits",
        "now64",
        "yesterday",
        "nowinblock",
        "nowinblock64",
        "uptime",
        "queryid",
        "currentqueryid",
        "initialqueryid",
        "initialquerystarttime",
        "transactionid",
        "generaterandomstructure",
        "filesystemavailable",
        "filesystemfree",
        "filesystemcapacity",
        # Snowflake
        "uuid_string",
        "seq1",
        "seq2",
        "seq4",
        "seq8",
        "randstr",
    }
)
_VOLATILE_NODES = tuple(
    node
    for node in (
        getattr(exp, name, None)
        for name in (
            "Rand",
            "Randn",
            "Uuid",
            "Seq1",
            "Seq2",
            "Seq4",
            "Seq8",
            # `now()`, `CURRENT_TIMESTAMP`, `SYSDATE`, `GETDATE()`, ...
            "CurrentDate",
            "CurrentDatetime",
            "CurrentTime",
            "CurrentTimestamp",
            "CurrentTimestampLTZ",
            "Localtime",
            "Localtimestamp",
            "Systimestamp",
            "UtcDate",
            "UtcTime",
            "UtcTimestamp",
        )
    )
    if node is not None
)


def _is_volatile(node: Any, dialect: str | None, relations: set[str]) -> bool:
    if isinstance(node, _VOLATILE_NODES):
        return True
    if (
        dialect == "sqlite"
        and _function_name(node) in _SQLITE_TIME_FUNCTIONS
        and not any(True for _ in node.iter_expressions())
    ):
        # `datetime()`, `unixepoch()`: without a time value, the current time
        return True
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        if name == "unix_timestamp":
            # of its argument, if it has one
            return not node.expressions
        return name in _VOLATILE_FUNCTIONS or (
            dialect == "clickhouse" and name.startswith("rand")
        )
    if isinstance(node, exp.Literal) and node.is_string:
        return _is_current_time_literal(node)
    if isinstance(node, exp.Dot):
        # Oracle `DBMS_RANDOM.RANDOM()`, `DBMS_CRYPTO.RANDOMBYTES(…)`
        return node.this.name.lower() in ("dbms_random", "dbms_crypto")
    if isinstance(node, exp.Column) and node.table:
        # `seq.NEXTVAL` (Oracle, Snowflake), `DBMS_RANDOM.VALUE` without parentheses; a
        # qualifier that names a table or alias of the query is a column `nextval`
        qualifier = node.table.lower()
        if qualifier in relations:
            return False
        return node.name.lower() == "nextval" or qualifier in (
            "dbms_random",
            "dbms_crypto",
        )
    return False


# Text that names a time relative to now: Postgres' special date / time inputs, SQLite's
# 'now' time value
_CURRENT_TIME_TEXT = frozenset({"now", "today", "tomorrow", "yesterday"})
# SQLite's date and time functions, whose time value may be 'now' (or left out)
_SQLITE_TIME_FUNCTIONS = frozenset(
    {"date", "time", "datetime", "julianday", "unixepoch", "strftime", "timediff"}
)
# Functions that read their text argument as a time: sqlglot's names for the SQLite ones
# above and their kin (`datetime('now')` parses as `TS_OR_DS_TO_TIMESTAMP`), by name
_TIME_FUNCTIONS = _SQLITE_TIME_FUNCTIONS | {
    "ts_or_ds_to_timestamp",
    "ts_or_ds_to_date",
    "time_to_str",
    "str_to_time",
    "timestamp",
    "to_timestamp",
    "to_date",
}


def _function_name(node: Any) -> str:
    if isinstance(node, (exp.Anonymous, exp.AnonymousAggFunc)):
        return str(node.name).lower()
    if isinstance(node, exp.Func):
        return str(node.sql_name()).lower()
    return ""


# Where `'now'` is only text: concatenated, a `LIKE` pattern, an output column of its
# own
_TEXT_CONTEXTS = (exp.DPipe, exp.Concat, exp.ConcatWs, exp.Like, exp.ILike, exp.Alias)
# String functions, by sqlglot's name or the function's own
_TEXT_FUNCTIONS = frozenset(
    {
        "instr",
        "replace",
        "str_position",
        "strpos",
        "position",
        "locate",
        "substring",
        "substr",
        "left",
        "right",
        "lower",
        "upper",
        "trim",
        "ltrim",
        "rtrim",
        "length",
        "char_length",
        "split_part",
        "starts_with",
        "ends_with",
        "contains",
        "lpad",
        "rpad",
        "repeat",
        "reverse",
        "format",
        "regexp_like",
        "regexp_replace",
    }
)


def _is_current_time_literal(node: Any) -> bool:
    """
    A text literal that the database may read as the current time: `'now'::timestamp`,
    SQLite's `datetime('now')`, and wherever its type comes from what it meets (Postgres
    reads `ts - 'now'`, `created < 'now'` and `greatest(ts, 'now')` as times). Only
    where it is clearly text (`s || 'now'`, a cast to text) is it not one.
    """
    if node.name.strip().lower() not in _CURRENT_TIME_TEXT:
        return False
    parent = node.parent
    if isinstance(parent, (exp.Cast, exp.TryCast)):
        return bool(parent.to.is_type(*exp.DataType.TEMPORAL_TYPES))
    return not isinstance(parent, (*_TEXT_CONTEXTS, exp.Select)) and (
        _function_name(parent) not in _TEXT_FUNCTIONS
    )


# Aggregates that return any one of their values; ClickHouse's numbering of the rows
# (and blocks) in the order they happen to be read
_ARBITRARY_FUNCTIONS = frozenset(
    {
        "any",
        "anylast",
        "any_value",
        "arbitrary",
        "first",
        "last",
        "rownumberinallblocks",
        "rownumberinblock",
        "blocknumber",
    }
)


# ClickHouse functions of the rows read before (their number, or values): a filter the
# database runs before them, also on another column, changes their values
_ROWS_READ_FUNCTIONS = frozenset(
    {
        "rownumberinallblocks",
        "rownumberinblock",
        "blocknumber",
        "runningdifference",
        "runningdifferencestartingwithfirstvalue",
        "runningaccumulate",
        "neighbor",
        # the size of the block read
        "blocksize",
        "blockserializedsize",
    }
)


# Window functions whose value depends on the order of rows that tie (`row_number() OVER
# ()`; an order is taken as not unique); rank(), and aggregates over the default frame,
# treat ties alike
_ROW_ORDER_WINDOWS = frozenset(
    {"row_number", "ntile", "first_value", "last_value", "nth_value", "lag", "lead"}
)


def _picks_arbitrary_rows(stmt: Any, dialect: str | None) -> bool:
    """
    Whether the query may give other rows, or number them otherwise, each time it runs:
    a window function that tells tied rows apart (or has a `ROWS` / `GROUPS` frame),
    `DISTINCT ON`, an aggregate that picks any value, Oracle's `ROWNUM`, or Postgres'
    one-argument `age(x)` (relative to today).
    """
    if stmt.find(exp.AnyValue, exp.First, exp.Last) is not None:
        return True
    for window in stmt.find_all(exp.Window):
        spec = window.args.get("spec")
        kind = str(spec.args.get("kind") or "").lower() if spec is not None else ""
        if kind in ("rows", "groups") or (
            _function_name(window.this) in _ROW_ORDER_WINDOWS
            or type(window.this).__name__.lower()
            in (
                "rownumber",
                "ntile",
                "firstvalue",
                "lastvalue",
                "nthvalue",
                "lag",
                "lead",
            )
        ):
            return True
    if dialect == "oracle" and any(
        c.name.lower() == "rownum" and not c.table for c in stmt.find_all(exp.Column)
    ):
        return True
    if any(d.args.get("on") is not None for d in stmt.find_all(exp.Distinct)):
        return True
    for node in stmt.find_all(exp.Func):
        name = _function_name(node)
        # each partition reads other blocks, and other rows before each
        if name in _ARBITRARY_FUNCTIONS or name in _ROWS_READ_FUNCTIONS:
            return True
        if name == "age" and len(node.expressions) == 1:
            return True
    return False


def _top_level_order_by_without_limit(stmt: Any) -> bool:
    """
    An `ORDER BY` without a `LIMIT` at the query's top level, also inside parentheses
    (`(SELECT … ORDER BY …)`): a database may drop it inside a derived table.
    """
    while True:
        has_limit = any(
            stmt.args.get(k) is not None for k in ("limit", "fetch", "offset")
        )
        if stmt.args.get("order") is not None and not has_limit:
            return True
        if not isinstance(stmt, (exp.Subquery, exp.Paren)):
            return False
        stmt = stmt.this


def _nested_queries(stmt: Any) -> bool:
    """
    A CTE, a query inside the query (one not reached from the top only through
    parentheses and set operations, `UNION`), or a parenthesised one with a `LIMIT` of
    its own: MySQL may evaluate any of them when it plans the query.
    """
    if stmt.find(exp.CTE) is not None:
        return True
    top = [stmt]
    top_selects = 0
    while top:
        node = top.pop()
        if node is not stmt and any(
            node.args.get(k) for k in ("limit", "fetch", "offset")
        ):
            return True
        if isinstance(node, exp.Select):
            top_selects += 1
        elif isinstance(node, (exp.Subquery, exp.Paren)):
            top.append(node.this)
        elif isinstance(node, exp.SetOperation):
            top.extend((node.this, node.expression))
    return sum(1 for _ in stmt.find_all(exp.Select)) > top_selects


def _relations(stmt: Any) -> set[str]:
    """Lower-cased names and aliases of the tables, subqueries and CTEs of a query."""
    names: set[str] = set()
    for node in stmt.find_all(exp.Table, exp.Subquery, exp.CTE):
        names.add(node.alias_or_name.lower())
        if isinstance(node, exp.Table):
            names.add(node.name.lower())
    names.discard("")
    return names


# Where a CTE that names itself is recursive without `RECURSIVE`
_IMPLICITLY_RECURSIVE = frozenset({"sqlite", "oracle", "tsql"})


def _cte_in_scope(table: Any, dialect: str | None) -> bool:
    """
    Whether a CTE of the query (or of an enclosing one) visible where the table is named
    has its name. A CTE is visible in the query's body and in the CTEs after it, not in
    its own definition (`WITH v AS (SELECT * FROM v)` reads the table `v`), unless
    `RECURSIVE` (or where a CTE naming itself is recursive anyway: SQLite, Oracle,
    SQL Server).
    """
    if table.args.get("db"):
        return False
    # _tables has already normalized identifiers according to the dialect, including
    # its treatment of quoted names (SQL Server's stay as written). Folding again would
    # conflate Postgres v and "V".
    name = table.name
    path: list[Any] = [table]
    node = table.parent
    while node is not None:
        with_ = node.args.get("with_") or node.args.get("with")
        if with_ is not None:
            ctes = list(with_.expressions)
            visible = ctes
            if len(path) >= 2 and path[-1] is with_ and not with_.args.get("recursive"):
                # named inside the definition of one of them
                inside = path[-2]
                own = 1 if dialect in _IMPLICITLY_RECURSIVE else 0
                visible = next(
                    (ctes[: i + own] for i, c in enumerate(ctes) if c is inside), ctes
                )
            if any(c.alias_or_name == name for c in visible):
                return True
        path.append(node)
        node = node.parent
    return False


def _tables(stmt: Any, dialect: str | None) -> list[list[str]]:
    """
    The tables (or views) a query reads, as `[catalog.][schema.]name` parts, unquoted
    names folded as the dialect folds them (`Orders` is `orders` on Postgres, `ORDERS`
    on Oracle): not its CTEs. A table function may be a view too (ClickHouse's
    parameterized `v(p = 1)`).
    """
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

    # SQL Server's names fold by the collation of the database, case-sensitive on some:
    # sqlglot lower-cases them all, quoted ones too, so `V` (a view) and `v` (a CTE)
    # would conflate, and the catalog would look up `v`. They stay as written: the
    # catalog lookup compares them as the database does, and a CTE hides only a name
    # of its exact case (a case-insensitive database may look up a name its CTE hides).
    if dialect == "tsql":
        stmt = stmt.copy()
    else:
        stmt = normalize_identifiers(stmt.copy(), dialect=dialect)
    tables: list[list[str]] = []
    for t in stmt.find_all(exp.Table):
        if isinstance(t.this, exp.Identifier):
            name = t.name
        elif isinstance(t.this, exp.Anonymous):
            name = str(t.this.name)
        else:
            continue
        if not name or _cte_in_scope(t, dialect):
            continue
        parts = [p for p in (t.catalog, t.db, name) if p]
        if parts not in tables:
            tables.append(parts)
    return tables


# Where the catalog tells a user's table function (a DuckDB table macro running
# `random()`, a volatile Postgres set-returning function) from the stable built-ins
_TABLE_FUNCTION_CATALOGS = frozenset({"duckdb", "postgres", "clickhouse"})


def _unknown_functions(stmt: Any, dialect: str | None) -> list[str]:
    """
    Lower-cased names of the functions sqlglot doesn't know; table functions only where
    the catalog tells user ones from built-ins (`_TABLE_FUNCTION_CATALOGS`).
    """
    tables = dialect in _TABLE_FUNCTION_CATALOGS
    return sorted(
        {
            str(n.name).lower()
            for n in stmt.find_all(exp.Anonymous)
            if n.name and (tables or not isinstance(n.parent, exp.Table))
        }
    )


# Where the user can replace a built-in function (SQLite's `create_function`, a DuckDB
# macro or Python function), so that the name sqlglot knows runs something else
_OVERRIDABLE_DIALECTS = frozenset({"sqlite", "duckdb"})


# The functions that run for an operator the user can replace, by the operator's token
_OPERATOR_FUNCTIONS = {
    "sqlite": {"regexp": "regexp", "match": "match"},
    "duckdb": {
        # (`NOT LIKE` runs `!~~`; `SIMILAR TO` is one token)
        "like": "~~",
        "ilike": "~~*",
        "glob": "~~~",
        "similar to": "regexp_full_match",
        "extract": "date_part",
        "//": "//",
        "%": "%",
        "-": "-",
        "^": "^",
        "&": "&",
        "<<": "<<",
        "&&": "&&",
        "->>": "->>",
        "@": "@",
        "||": "||",
        "/": "/",
        ">>": ">>",
        "|": "|",
        "**": "**",
        "^@": "^@",
        "@>": "@>",
    },
}


def _known_functions(
    stmt: Any | None, tokens: list[Any], dialect: str | None
) -> list[str]:
    """
    Lower-cased names of the functions of the query that a user's function of the name
    may replace (`_OVERRIDABLE_DIALECTS`): the database says if one does. Those sqlglot
    knows under another name (`instr`, `ltrim`), as written, and `LIKE` / `GLOB`, which
    SQLite runs as the functions of the name. `stmt` is `None` for a query sqlglot
    doesn't parse: the names its tokens show only.
    """
    if dialect not in _OVERRIDABLE_DIALECTS:
        return []
    names: set[str] = set()
    if stmt is not None:
        for f in stmt.find_all(exp.Func):
            if not isinstance(f, exp.Anonymous):
                names.update(n.lower() for n in f.sql_names() if n.isidentifier())
    operators = _OPERATOR_FUNCTIONS.get(dialect, {})
    for i, t in enumerate(tokens):
        word = t.text.lower()
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if nxt is not None and nxt.token_type == sqlglot.TokenType.L_PAREN:
            # (not a keyword: `IN (`, `FROM (`, `AS (`, `OVER (`)
            if t.text.isidentifier() and t.token_type in (
                sqlglot.TokenType.VAR,
                sqlglot.TokenType.IDENTIFIER,
            ):
                names.add(word)
            if word == "count" and dialect == "duckdb" and i + 2 < len(tokens):
                if tokens[i + 2].token_type == sqlglot.TokenType.STAR:
                    names.add("count_star")
        if word in operators:
            names.add(operators[word])
            if dialect == "duckdb" and word in ("like", "ilike", "glob"):
                names.add("!" + operators[word])
        if dialect == "duckdb" and word == "escape":
            names.update(("like_escape", "not_like_escape", "ilike_escape"))
    if dialect == "sqlite":
        if stmt is None:
            names.update(
                t.text.lower() for t in tokens if t.text.lower() in ("like", "glob")
            )
        else:
            if stmt.find(exp.Like, exp.ILike) is not None:
                names.add("like")
            if stmt.find(exp.Glob) is not None:
                names.add("glob")
    # syntax, not a function a user's can replace
    names.discard("cast")
    return sorted(names)


def _volatile_columns(stmt: Any, calls: list[Any]) -> list[str] | None:
    """
    The output columns whose values come from volatile functions, `None` if that can't
    be told (all of them, then). A filter pushed on such a column is wrong where the
    database merges the wrapped query into the filter's and runs the function again for
    it (SQLite): `random() AS r` is then filtered on one value and returns another.
    """
    projections = (
        {id(p): p for p in stmt.selects} if isinstance(stmt, exp.Select) else {}
    )
    columns: list[str] = []
    for call in calls:
        node = call
        while node is not None and id(node) not in projections:
            if node is stmt:
                break
            node = node.parent
        if node is not None and id(node) in projections:
            if node.is_star or not node.alias_or_name:
                # `t.*`, or a name the database makes up (`random()`)
                return None
            columns.append(node.alias_or_name)
            continue
        if call.find_ancestor(exp.Select) is not stmt:
            # in a subquery or CTE, whose columns the outer query may return
            return None
        # a condition or ordering of the query itself (`WHERE random() < 0.5`) only
        # picks rows, as it would without the filter
    return columns


def _no_facts() -> dict[str, Any]:
    """The facts of a query that doesn't parse."""
    return {
        "parses": False,
        "single_select": False,
        "has_cte": False,
        "top_level_order_by_without_limit": False,
        "output_columns": None,
        "read_only": False,
        "repeatable": False,
        "top_level_limit": False,
        "nested_queries": False,
        "side_effects": False,
        "side_effects_outside_rows": False,
        "volatile_columns": [],
        "all_columns_volatile": False,
        "unknown_functions": [],
        "known_functions": [],
        "tables": [],
        "body": None,
    }


def _without_lambda_parentheses(sql: str, dialect: str | None) -> str:
    """
    DuckDB writes the lambdas of a view (and of a list comprehension) parenthesised,
    `list_transform(l, (lambda x: (x + 1)))`, which sqlglot doesn't parse: the same SQL
    with blanks for those parentheses, which it does. Only a parenthesis that opens an
    argument (after `(` or `,`), not a call's.
    """
    if dialect != "duckdb" or "lambda" not in sql.lower():
        return sql
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    except sqlglot.errors.SqlglotError:
        return sql
    tt = sqlglot.TokenType
    blanks: list[int] = []
    opened: list[int] = []
    for i, t in enumerate(tokens):
        if t.token_type == tt.L_PAREN:
            opened.append(i)
        elif t.token_type == tt.R_PAREN and opened:
            j = opened.pop()
            if (
                j >= 1
                and j + 1 < i
                and tokens[j - 1].token_type in (tt.L_PAREN, tt.COMMA)
                and tokens[j + 1].token_type == tt.VAR
                and tokens[j + 1].text.lower() == "lambda"
            ):
                blanks += [tokens[j].start, t.start]
    chars = list(sql)
    for at in blanks:
        chars[at] = " "
    return "".join(chars)


def _plain_placeholders(sql: str, dialect: str | None) -> str:
    """
    A query's parameter placeholders that sqlglot doesn't parse in its dialect (`%s` and
    `%(name)s` on MySQL, `:1` on Oracle) as `?`, after blanks: the same length.
    """
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    except sqlglot.errors.SqlglotError:
        return sql
    tt = sqlglot.TokenType
    chars = list(sql)
    for i, t in enumerate(tokens):
        after = tokens[i + 1 : i + 5]
        # (adjacent tokens: `% s` is a modulo)
        kinds = [a.token_type for a in after]
        adjacent = all(
            b.start == a.end + 1 for a, b in zip([t, *after], after, strict=False)
        )
        end = None
        if t.token_type == tt.MOD and kinds[:1] == [tt.VAR] and after[0].text == "s":
            end = after[0].end if after[0].start == t.end + 1 else None
        elif (
            t.token_type == tt.MOD
            and kinds == [tt.L_PAREN, tt.VAR, tt.R_PAREN, tt.VAR]
            and after[3].text == "s"
            and adjacent
        ):
            end = after[3].end
        elif t.token_type == tt.COLON and kinds[:1] == [tt.NUMBER]:
            end = after[0].end if after[0].start == t.end + 1 else None
        if end is not None:
            # (ending where the placeholder ends, as the query's last token may)
            chars[t.start : end + 1] = " " * (end - t.start) + "?"
    return "".join(chars)


def _parse_query(sql: str, dialect: str | None) -> tuple[list[Any], list[Any]]:
    """A query's statements and tokens: a comment after its last `;` is none."""
    statements = [
        s
        for s in sqlglot.parse(sql, read=dialect)
        if s is not None and not isinstance(s, exp.Semicolon)
    ]
    return statements, sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)


def _unparsed_facts(
    facts: dict[str, Any], sql: str, dialect: str | None
) -> dict[str, Any]:
    """
    The facts of a query sqlglot doesn't parse, with the functions its tokens call (a
    user's volatile macro may be among them: the query is partitioned all the same).
    With no tokens either, none.
    """
    try:
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
    except sqlglot.errors.SqlglotError:
        return facts
    facts["known_functions"] = _known_functions(None, tokens, dialect)
    return facts


def inspect_query(sql: str, dialect: str | None) -> dict[str, Any]:
    """Facts about a user query that decide whether it can be wrapped."""
    facts = _no_facts()
    # (the same length: its tokens are where the query's are)
    original, sql = sql, _without_lambda_parentheses(sql, dialect)
    try:
        statements, tokens = _parse_query(sql, dialect)
    except sqlglot.errors.SqlglotError:
        # a query's placeholders for `params=` (or the views it reads, and the functions
        # it calls, would go unchecked)
        plain = _plain_placeholders(sql, dialect)
        if plain == sql:
            return _unparsed_facts(facts, sql, dialect)
        try:
            statements, tokens = _parse_query(plain, dialect)
        except sqlglot.errors.SqlglotError:
            return _unparsed_facts(facts, sql, dialect)
        sql = plain
    facts["parses"] = True
    # the query without the semicolons and comments after its last token, which can't
    # go inside `(...)` (MySQL's `#` comment would comment out the parenthesis)
    while tokens and tokens[-1].token_type == sqlglot.TokenType.SEMICOLON:
        tokens.pop()
    if tokens and len(statements) == 1 and "/*!" not in sql[tokens[-1].end + 1 :]:
        # (a MySQL executable comment after it is part of the query)
        body = sql[: tokens[-1].end + 1]
        with contextlib.suppress(sqlglot.errors.SqlglotError):
            # only what the tokens say is the same query
            if sqlglot.parse(body, read=dialect) == statements:
                facts["body"] = original[: tokens[-1].end + 1]
    facts["read_only"] = len(statements) == 1 and _is_read_only(statements[0])
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return facts

    stmt = statements[0]
    facts["single_select"] = True
    facts["has_cte"] = stmt.args.get("with_") is not None or (
        stmt.args.get("with") is not None
    )
    facts["top_level_order_by_without_limit"] = _top_level_order_by_without_limit(stmt)
    # what can't take another `LIMIT` after it
    facts["top_level_limit"] = any(
        stmt.args.get(k) for k in ("limit", "fetch", "offset", "locks")
    )
    facts["nested_queries"] = _nested_queries(stmt)
    relations = _relations(stmt)
    volatile_calls = [
        node
        for node in stmt.find_all(exp.Func, exp.Dot, exp.Column, exp.Literal)
        if _is_volatile(node, dialect, relations)
    ]
    volatile = bool(volatile_calls)
    # a function sqlglot doesn't know may be the user's own: SQLite's `create_function`
    # makes non-deterministic ones by default (Postgres doesn't push a filter through a
    # volatile one)
    unknown = list(stmt.find_all(exp.Anonymous)) if dialect == "sqlite" else []
    columns = _volatile_columns(stmt, volatile_calls + unknown)
    if columns is None or any(
        _function_name(f) in _ROWS_READ_FUNCTIONS for f in stmt.find_all(exp.Func)
    ):
        facts["all_columns_volatile"] = True
    else:
        facts["volatile_columns"] = columns
    # The database says whether these are volatile (`catalog::volatile_functions`): a
    # user's function may be (`random()` inside), and each partition would run it again.
    facts["unknown_functions"] = _unknown_functions(stmt, dialect)
    facts["known_functions"] = _known_functions(stmt, tokens, dialect)
    # A view among them may call a volatile function (`catalog::view_definition`).
    facts["tables"] = _tables(stmt, dialect)
    # anywhere in the query: a nested LIMIT without a unique ORDER BY picks rows too
    facts["repeatable"] = (
        stmt.find(exp.Limit, exp.Fetch, exp.Offset, exp.TableSample) is None
        and not volatile
        and not _picks_arbitrary_rows(stmt, dialect)
    )
    # Running it may change something: a volatile function (`nextval`), or one sqlglot
    # doesn't know, which may be the user's own (a stored function that writes).
    facts["side_effects"] = volatile or stmt.find(exp.Anonymous) is not None
    # Where MariaDB evaluates such a function under a `LIMIT 0`: a set operation's
    # branches, a query without FROM, and conditions (`WHERE f() = 1` is constant)
    calls = [
        node
        for node in stmt.find_all(exp.Func, exp.Dot, exp.Column, exp.Literal)
        if isinstance(node, exp.Anonymous) or _is_volatile(node, dialect, relations)
    ]
    facts["side_effects_outside_rows"] = bool(calls) and (
        isinstance(stmt, exp.SetOperation)
        or stmt.args.get("from_", stmt.args.get("from")) is None
        or any(c.find_ancestor(exp.Where, exp.Having, exp.Join) for c in calls)
    )
    selects = stmt.selects
    if selects and not any(isinstance(s, exp.Star) or s.is_star for s in selects):
        facts["output_columns"] = [s.alias_or_name for s in selects]
    return facts


def inspect_view(sql: str, dialect: str | None) -> dict[str, Any]:
    """
    `inspect_query`'s facts about a view's query, from its definition: the whole `CREATE
    VIEW` statement or only its query. Its output columns (and volatile ones) are the
    view's, named by its column list if it has one.
    """
    sql = _without_lambda_parentheses(sql, dialect)
    try:
        statements = [
            s
            for s in sqlglot.parse(sql, read=dialect)
            if s is not None and not isinstance(s, exp.Semicolon)
        ]
    except sqlglot.errors.SqlglotError:
        statements = []
    if len(statements) != 1:
        return _no_facts()
    stmt = statements[0]
    names: list[str] | None = None
    if isinstance(stmt, exp.Create):
        if str(stmt.args.get("kind") or "").upper() != "VIEW" or not isinstance(
            stmt.expression, exp.Query
        ):
            return _no_facts()
        if isinstance(stmt.this, exp.Schema):
            names = [c.name for c in stmt.this.expressions]
        stmt = stmt.expression
    named = names is None
    if (
        names is not None
        and isinstance(stmt, exp.Select)
        and len(stmt.selects) == len(names)
        and not any(s.is_star for s in stmt.selects)
    ):
        # `CREATE VIEW v (a, b) AS SELECT x, random() …`: the view's names, by position
        stmt.set(
            "expressions",
            [
                exp.alias_(s.unalias(), exp.to_identifier(n, quoted=True))
                for s, n in zip(stmt.selects, names, strict=True)
            ],
        )
        named = True
    try:
        query = stmt.sql(dialect=dialect)
    except sqlglot.errors.SqlglotError:
        return _no_facts()
    facts = inspect_query(query, dialect)
    if not named:
        # its names aren't the query's: a volatile column may be any of them
        facts["output_columns"] = names
        if facts["volatile_columns"]:
            facts["volatile_columns"] = []
            facts["all_columns_volatile"] = True
    return facts
