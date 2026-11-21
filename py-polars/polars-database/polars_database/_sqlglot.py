"""
The sqlglot SQL renderer behind the Rust `SqlRenderer` trait.

Rust sends a dialect-neutral `SqlQuery` as JSON; this module builds sqlglot
expressions from it and generates SQL for the target dialect. It is the only code
that imports sqlglot, and it only runs on the machine that plans the query.
"""

from __future__ import annotations

import contextlib
import json
import re
import secrets
from typing import Any

from polars._dependencies import import_optional
from polars_database._builtin_functions import BUILTIN_FUNCTIONS, SYNTAX_WORDS

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
# ... and have `TIMESTAMP_NTZ` and `TIME` types (Hive has neither)
_SPARK_TIME_TYPES = {"databricks", "spark"}


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
            if x["with_tz"]:
                # an explicit offset: the instant, whatever the session's time zone
                if dialect in _SPARK_TIME_TYPES:
                    # `TIMESTAMP` is NTZ where `spark.sql.timestampType` says so
                    return _raw(f"CAST('{x['text']}+00:00' AS TIMESTAMP_LTZ)")
                return _cast(x["text"] + "+00:00", "TIMESTAMP")
            if dialect in _SPARK_TIME_TYPES:
                # `TIMESTAMP` is the local-time-zone type: an NTZ column compared with
                # it is converted in the session's zone, which skips or repeats times
                return _ansi("TIMESTAMP_NTZ", x["text"])
            return _cast(x["text"], "TIMESTAMP")
        if x["with_tz"]:
            return _cast(x["text"] + "+00:00", _TIMESTAMPTZ)
        return _cast(x["text"], _TIMESTAMP)
    if kind == "Time":
        if dialect in _SPARK_TIME_TYPES:
            # sqlglot writes a Spark `TIME` as `TIMESTAMP`
            return _ansi("TIME", x)
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


def _finish(sql: str, query: dict[str, Any]) -> str:
    """
    The generated SQL with the user's query in place of its placeholder. Under `params=`
    the statement is %-formatted by the driver: a `%` the plugin wrote (a column's name)
    is doubled, the user's query is left as they wrote it.
    """
    if query.get("escape_percent"):
        sql = sql.replace("%", "%%")
    from_ = query["from"]
    if "Subquery" in from_:
        sub = from_["Subquery"]
        # a newline ends a trailing `--` (or MySQL `#`) comment of the user's query
        end = "\n" if "--" in sub or "#" in sub else ""
        sql = sql.replace(_SUBQUERY_PLACEHOLDER, f"({sub}{end})", 1)
    return sql


def render(query_json: str, dialect: str | None) -> str:
    """Render a `SqlQuery` (as JSON) for a dialect (`None` is sqlglot's default)."""
    query = json.loads(query_json)
    sql = str(_escaped_identifiers(build(query, dialect), dialect).sql(dialect=dialect))
    return _finish(sql, query)


_PREDICATE_PLACEHOLDER = f"__polars_predicate_{secrets.token_hex(8)}__"
_FILTER_PLACEHOLDER = f"__polars_filter_{secrets.token_hex(8)}__"


def _shared_filter(query: dict[str, Any], d: str | None) -> tuple[str, str]:
    """
    The `AND` conjuncts of a query's filter as SQL, as `WHERE` writes them: with
    another conjunct beside them (each of `AND` / `OR` in parentheses), and alone (a
    single conjunct isn't wrapped).
    """
    columns = set(query.get("qualified") or ())
    plain, wrapped = [], []
    for f in query["filter"]:
        node = _qualified(_expr(f, d), columns, query["from"])
        text = str(_escaped_identifiers(node, d).sql(dialect=d))
        plain.append(text)
        wrapped.append(f"({text})" if isinstance(node, exp.Connector) else text)
    return " AND ".join(wrapped), " AND ".join(plain)


def _with_partition(
    query: dict[str, Any], predicate: Any, shared: tuple[str, str], d: str | None
) -> str:
    """
    `query` with the partition's predicate ahead of its (already rendered) filter. The
    filter (an `IN` list of thousands of values, say) is written once for all
    partitions.
    """
    q = dict(query)
    # the partition predicates written by the user (`partition_predicates`, the only raw
    # ones) stay as written, also where `%` is doubled: they stand in as a placeholder
    user_text = None
    if isinstance(predicate, dict) and "Raw" in predicate:
        user_text, predicate = predicate["Raw"], {"Raw": _PREDICATE_PLACEHOLDER}
    filters = [] if predicate is None else [predicate]
    if not query["filter"]:
        q["filter"] = filters
        sql = str(_escaped_identifiers(build(q, d), d).sql(dialect=d))
        sql = _finish(sql, query)
    else:
        # the shared filter stands in as one raw condition, replaced after generation
        q["filter"] = [*filters, {"Raw": _FILTER_PLACEHOLDER}]
        sql = str(_escaped_identifiers(build(q, d), d).sql(dialect=d))
        alone = not filters and len(query["filter"]) == 1
        sql = sql.replace(f"({_FILTER_PLACEHOLDER})", shared[1 if alone else 0], 1)
        sql = _finish(sql, query)
    if user_text is not None:
        sql = sql.replace(_PREDICATE_PLACEHOLDER, user_text, 1)
    return sql


def render_partitions(
    query_json: str, count_json: str, predicates_json: str, dialect: str | None
) -> list[tuple[str, str]]:
    """
    The SQL and the row-count SQL of each partition: the query (and count query), each
    with the partition's predicate in front of its filter, which every partition shares.
    """
    query, count = json.loads(query_json), json.loads(count_json)
    shared = _shared_filter(query, dialect)
    if count["filter"] == query["filter"] and count.get("qualified") == query.get(
        "qualified"
    ):
        shared_count = shared
    else:
        shared_count = _shared_filter(count, dialect)
    return [
        (
            _with_partition(query, predicate, shared, dialect),
            _with_partition(count, predicate, shared_count, dialect),
        )
        for predicate in json.loads(predicates_json)
    ]


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


# ALLOWLIST OF REPEATABLE CONSTRUCTS
#
# Design: a query is repeatable only on positive proof that running it again returns the
# same rows (listing what is volatile kept missing constructs). Every function,
# operator, cast, column and table in it must be in the lists below (all small and
# conservative, per dialect where they differ); anything else -- a function sqlglot
# doesn't know, a user's function whatever its IMMUTABLE / DETERMINISTIC label, a clock,
# a session variable, a table function, a typed node nobody listed -- is *not*
# repeatable. A refused case costs a clear error (explicit `partition_on`) or one query
# (`partition_on="auto"`); a case wrongly proven returns wrong rows, so a name is added
# only when it is known to be deterministic. What the SQL can't show (a view's body, a
# table's kind, a generated column, a user's overload of a built-in) is proven on the
# Rust side from the catalog (`catalog.rs`, `planner.rs::source_volatility`) by the same
# rule.

# sqlglot's typed functions, by class name, whose value depends on their arguments only.
# (Not: clocks, random and uuid functions, `ANY_VALUE` / `FIRST` / `LAST` / `ARG_MIN`
# and kin, `ARRAY_AGG` / `GROUP_CONCAT` / `LIST` (order-dependent), `MODE`, approximate
# aggregates, `ROW_NUMBER` / `LAG` / `FIRST_VALUE` ... (order of ties), session and
# environment functions, JSON and array functions.)
_PROVEN_FUNCTIONS = frozenset(
    {
        # aggregates that don't depend on the order of rows
        "Count",
        "CountIf",
        "Sum",
        "Avg",
        "Min",
        "Max",
        "Stddev",
        "StddevPop",
        "StddevSamp",
        "Variance",
        "VariancePop",
        "LogicalAnd",
        "LogicalOr",
        "BitwiseAndAgg",
        "BitwiseOrAgg",
        "BitwiseXorAgg",
        "Corr",
        "CovarPop",
        "CovarSamp",
        "Median",
        "PercentileCont",
        "PercentileDisc",
        "Kurtosis",
        "Skewness",
        "Grouping",
        # windows that treat ties alike
        "Rank",
        "DenseRank",
        "PercentRank",
        "CumeDist",
        # conditions and conversions
        "And",
        "Or",
        "Xor",
        "Case",
        "If",
        "Coalesce",
        "Nullif",
        "Nvl2",
        "Greatest",
        "Least",
        "Cast",
        "TryCast",
        "Exists",
        "Decode",
        "DecodeCase",
        "Array",
        # math
        "Abs",
        "Acos",
        "Asin",
        "Atan",
        "Atan2",
        "Cbrt",
        "Ceil",
        "Cos",
        "Cot",
        "Degrees",
        "Exp",
        "Floor",
        "Ln",
        "Log",
        "Pi",
        "Pow",
        "Radians",
        "Round",
        "Sign",
        "Sin",
        "Sqrt",
        "Tan",
        "Trunc",
        "Factorial",
        # strings
        "Concat",
        "ConcatWs",
        "Contains",
        "EndsWith",
        "StartsWith",
        "Initcap",
        "Left",
        "Right",
        "Length",
        "Lower",
        "Upper",
        "Pad",
        "Repeat",
        "Replace",
        "Reverse",
        "Space",
        "SplitPart",
        "StrPosition",
        "Substring",
        "SubstringIndex",
        "Translate",
        "Trim",
        "Overlay",
        "Ascii",
        "Chr",
        "Unicode",
        "Hex",
        "Unhex",
        "MD5",
        "SHA",
        "SHA2",
        "Levenshtein",
        "Soundex",
        "ByteLength",
        "BitLength",
        "RegexpLike",
        "RegexpILike",
        "RegexpReplace",
        "RegexpExtract",
        "RegexpSubstr",
        "RegexpCount",
        "RegexpInstr",
        "ToChar",
        "ToNumber",
        "ToDouble",
        "ToBoolean",
        "NumberToStr",
        # dates and times of their arguments (a text 'now' is
        # `_is_current_time_literal`; SQLite's without a time value,
        # `_sqlite_time_unproven`)
        "Extract",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Second",
        "Quarter",
        "Week",
        "WeekOfYear",
        "DayOfWeek",
        "DayOfMonth",
        "DayOfYear",
        "DateAdd",
        "DateSub",
        "DateDiff",
        "DateTrunc",
        "DatetimeTrunc",
        "TimestampTrunc",
        "DateFromParts",
        "TimeFromParts",
        "TimestampFromParts",
        "LastDay",
        "AddMonths",
        "MonthsBetween",
        "TsOrDsAdd",
        "TsOrDsDiff",
        "TsOrDsToDate",
        "TsOrDsToTimestamp",
        "TsOrDsToDatetime",
        "TsOrDsToTime",
        "StrToDate",
        "StrToTime",
        "TimeToStr",
        "UnixToTime",
        "UnixToStr",
        "TimeStrToTime",
        "TimeStrToDate",
        "Date",
        "Datetime",
        "Time",
        "Timestamp",
    }
)

# Functions sqlglot doesn't know (parsed as `Anonymous`) that are built in and
# deterministic, lower-cased, in the dialects below (a user's function of the name is
# looked up in the catalog: `_OVERRIDABLE_DIALECTS`).
_PROVEN_ANONYMOUS = frozenset(
    {
        "ifnull",
        "nvl",
        "iif",
        "typeof",
        "hex",
        "quote",
        "printf",
        "instr",
        "unicode",
        "octet_length",
        "char_length",
        "character_length",
        "btrim",
        "date_part",
        "every",
        "total",
    }
)
# ... and in one dialect
_PROVEN_ANONYMOUS_BY_DIALECT = {
    "clickhouse": frozenset(
        {
            "tostring",
            "toint8",
            "toint16",
            "toint32",
            "toint64",
            "touint8",
            "touint16",
            "touint32",
            "touint64",
            "tofloat32",
            "tofloat64",
            "todate",
            "todatetime",
            "todate32",
            "empty",
            "notempty",
            "intdiv",
            "modulo",
            "lowerutf8",
            "upperutf8",
            "lengthutf8",
            "multiif",
        }
    ),
}

# Spark table functions that return the same rows on every call (their arguments are
# checked as any); a user's table function, `read_kafka` or `IDENTIFIER('v')` is not
# proven
_SPARK_DIALECTS = frozenset({"databricks", "spark", "hive"})
_SPARK_TABLE_FUNCTIONS = frozenset(
    {
        "range",
        "explode",
        "explode_outer",
        "inline",
        "inline_outer",
        "posexplode",
        "posexplode_outer",
        "stack",
        "json_tuple",
    }
)

# Everything else the tree may hold: syntax that holds no function or table of its own.
_PROVEN_STRUCTURE = frozenset(
    {
        "Select",
        "Subquery",
        "Paren",
        "Tuple",
        "Where",
        "Prewhere",
        "From",
        "Join",
        "Having",
        "Group",
        "Cube",
        "Rollup",
        "GroupingSets",
        "Order",
        "Ordered",
        "Alias",
        "TableAlias",
        "Column",
        "Identifier",
        "Literal",
        "Null",
        "Boolean",
        "Star",
        "DataType",
        "DataTypeParam",
        "In",
        "Between",
        "Is",
        "Not",
        "Neg",
        "EQ",
        "NEQ",
        "LT",
        "LTE",
        "GT",
        "GTE",
        "NullSafeEQ",
        "NullSafeNEQ",
        "Add",
        "Sub",
        "Mul",
        "Div",
        "IntDiv",
        "Mod",
        "DPipe",
        "Like",
        "ILike",
        "Escape",
        "BitwiseAnd",
        "BitwiseOr",
        "BitwiseXor",
        "BitwiseNot",
        "BitwiseLeftShift",
        "BitwiseRightShift",
        "Union",
        "Intersect",
        "Except",
        "With",
        "CTE",
        "Distinct",
        "Window",
        "WindowSpec",
        "Filter",
        "WithinGroup",
        "Interval",
        "Var",
        "Values",
        "Any",
        "All",
        "Placeholder",
        "Table",
        "Dot",
        "Bracket",
        "Limit",
        "Offset",
        "Fetch",
        "TableSample",
        "Lateral",
        "Final",
    }
)

# Keywords a parser reads as a column (it knows none of them as a function) that name
# the time or the session, as an unqualified column: also a real column of the name is
# refused
_NILADIC_COLUMNS = frozenset(
    {
        "sysdate",
        "systimestamp",
        "current_date",
        "current_time",
        "current_timestamp",
        "localtime",
        "localtimestamp",
        "utc_date",
        "utc_time",
        "utc_timestamp",
        "sessiontimezone",
        "dbtimezone",
        "current_user",
        "session_user",
        "user",
        "uid",
        "current_catalog",
        "current_schema",
        "current_role",
        "current_path",
        "rownum",
        "now",
        "today",
    }
)


def _sqlite_time_unproven(node: Any) -> bool:
    """
    SQLite's date and time functions read their first argument as a time value: without
    one, or with a modifier alone (`datetime('subsec')`), or `'now'`, they return the
    current time. SQLite is dynamically typed, so a column or expression may hold
    `'now'`: proven only for a literal, a number or a text that starts with a digit.
    """
    first = next(iter(node.iter_expressions()), None)
    if first is None:
        return True
    if isinstance(first, exp.Literal):
        return first.is_string and not first.name[:1].isdigit()
    return True


# ClickHouse functions of the rows read before (their number, or values, or the size of
# the block): a filter the database runs before them, also on another column, changes
# them. About filter pushdown, not repeatability (those aren't proven anyway).
_ROWS_READ_FUNCTIONS = frozenset(
    {
        "rownumberinallblocks",
        "rownumberinblock",
        "blocknumber",
        "runningdifference",
        "runningdifferencestartingwithfirstvalue",
        "runningaccumulate",
        "neighbor",
        "blocksize",
        "blockserializedsize",
    }
)


def _describe(node: Any, dialect: str | None) -> str:
    """What an unproven node is, for the error that names it: its name, or its SQL."""
    if isinstance(node, exp.Func):
        name = _function_name(node)
        return f"{name}()" if name else "a function"
    if isinstance(node, exp.Table):
        return str(node.sql(dialect=dialect))[:40]
    try:
        return str(node.sql(dialect=dialect))[:40]
    except sqlglot.errors.SqlglotError:
        return type(node).__name__


def _oracle_clock_date(node: Any) -> bool:
    """
    Whether an Oracle `TO_DATE` / `TO_TIMESTAMP` may take its year or month from the
    clock: a format of no year or no month fills them in from `SYSDATE` (a Julian day
    has both; no format is the session's, which has both by default).
    """
    fmt = node.args.get("format")
    if fmt is None:
        return False
    if not isinstance(fmt, exp.Literal) or not fmt.is_string:
        return True
    text = fmt.name
    upper = text.upper()
    if upper.strip() == "J":
        return False
    year = any(t in text for t in ("%Y", "%y")) or "Y" in upper or "RR" in upper
    month = any(t in text for t in ("%m", "%b", "%B", "%j")) or any(
        t in upper for t in ("MM", "MON", "RM", "DDD")
    )
    return not (year and month)


def _unproven(node: Any, dialect: str | None, relations: set[str]) -> bool:
    """Whether `node` is something the allowlist doesn't prove to repeat."""
    kind = type(node).__name__
    if (
        dialect == "oracle"
        and isinstance(node, (exp.StrToDate, exp.StrToTime))
        and _oracle_clock_date(node)
    ):
        return True
    if (
        dialect in _SPARK_DIALECTS
        and isinstance(node.parent, exp.Table)
        and node.parent.this is node
        and isinstance(node, exp.Func)
    ):
        return _function_name(node) not in _SPARK_TABLE_FUNCTIONS
    if (
        isinstance(node, exp.Dot)
        and isinstance(node.expression, exp.Func)
        and dialect in _USER_ROUTINE_DIALECTS
    ):
        # `schema.f(x)`: a user's routine by a path the catalog lookup of names doesn't
        # follow
        return True
    if isinstance(node, exp.Anonymous):
        name = node.name.lower()
        if name == "unix_timestamp":
            # of its argument, if it has one
            return not node.expressions
        if name in _SQLITE_TIME_FUNCTIONS and dialect == "sqlite":
            return _sqlite_time_unproven(node)
        return not (
            name in _PROVEN_ANONYMOUS
            or name in _PROVEN_ANONYMOUS_BY_DIALECT.get(dialect or "", ())
        )
    if isinstance(node, exp.Join):
        return _row_picking_join(node, dialect)
    if isinstance(node, exp.Query) and _unsafe_settings(node):
        return True
    if isinstance(node, exp.Func):
        if kind not in _PROVEN_FUNCTIONS:
            return True
        if isinstance(node, exp.Array) and any(
            isinstance(e, exp.Query) for e in node.expressions
        ):
            # `ARRAY(SELECT ...)`: the order of the subquery's rows is none; only a
            # constructor of proven elements is
            return True
        if dialect in _TEXT_TIME_DIALECTS and _time_cast_operand(node) is not None:
            return _unproven_temporal_cast(node, dialect)
        if dialect == "sqlite" and _function_name(node) in _TIME_FUNCTIONS:
            return _sqlite_time_unproven(node)
        return False
    if isinstance(node, exp.DataType):
        # a user's type (Postgres): its casts and comparisons are the user's
        return bool(node.this == exp.DataType.Type.USERDEFINED)
    if isinstance(node, exp.Column):
        if node.is_star:
            return False
        name = node.name.lower()
        if node.table:
            # `seq.NEXTVAL`, `DBMS_RANDOM.VALUE`, a struct's field: a qualifier that is
            # no table, subquery or CTE of the query isn't proven to be a column
            if node.table.lower() not in relations:
                return True
        elif name in _NILADIC_COLUMNS:
            return True
        # ClickHouse's virtual columns (`_part`, `_partition_id`, ...)
        return dialect == "clickhouse" and name.startswith("_")
    if isinstance(node, exp.Table):
        # a name (a view or table the catalog proves: `catalog.view_definition`), not a
        # function, time travel, a partition clause or a sample
        is_name = isinstance(node.this, exp.Identifier) or (
            dialect in _SPARK_DIALECTS
            and isinstance(node.this, exp.Func)
            and _function_name(node.this) in _SPARK_TABLE_FUNCTIONS
        )
        return not is_name or any(
            node.args.get(k)
            for k in (
                "version",
                "system_time",
                "when",
                "changes",
                "sample",
                "rows_from",
            )
        )
    if isinstance(node, exp.Values) and dialect == "clickhouse":
        # `values('id UInt32, k UInt32 ALIAS modulo(rand(), 4)', ...)`: the structure
        # string is read as one literal, though its `ALIAS` / `DEFAULT` expressions run
        # on each read (the first row, whose first element is the structure, each in a
        # tuple)
        structure: Any = node
        for _ in range(3):
            exprs = structure.expressions if structure is not None else None
            structure = exprs[0] if exprs else None
        return (
            isinstance(structure, exp.Literal)
            and structure.is_string
            and re.search(
                r"\b(alias|default|materialized|ephemeral)\b", structure.name, re.I
            )
            is not None
        )
    if isinstance(node, exp.Literal) and node.is_string:
        return _is_current_time_literal(node)
    if isinstance(node, exp.Parameter):
        # `@v`, `@@v`: the session's (each partition reads on a session of its own); a
        # numbered `$1` is a bound parameter
        return not isinstance(node.this, exp.Literal)
    return kind not in _PROVEN_STRUCTURE


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


# Where a relative-time text is only text: concatenated, a `LIKE` pattern. (Not an
# output column of its own: `SELECT 'now' AS t` may be cast later, `(SELECT
# 'now')::timestamptz`.)
_TEXT_CONTEXTS = (exp.DPipe, exp.Concat, exp.ConcatWs, exp.Like, exp.ILike)
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


# A text a database may read as a time relative to now: a word of `_RELATIVE_WORDS` with
# nothing else but time-like tokens around it, in any order (`'today'`, `'yesterday
# 13:00'`, `'today EST'`, `'EST today'`, `'13:00 yesterday'`, `'today Europe/Paris'`,
# `'today allballs'`); not `'Now Corp'`. Mirrored by `planner.rs`'s
# `relative_time_text`.
_RELATIVE_WORDS = frozenset({"now", "today", "tomorrow", "yesterday"})
# Words of a time of day or a zone (Postgres' zone abbreviations and its zone names that
# are one word); a zone with a `/` is a path (`Europe/Paris`), `PST8PDT` or `UTC+5` a
# POSIX zone
_TIME_WORDS = frozenset(
    [
        "am",
        "pm",
        "at",
        "allballs",
        "z",
        "zulu",
        "ut",
        "utc",
        "uct",
        "gmt",
        "est",
        "edt",
        "cst",
        "cdt",
        "mst",
        "mdt",
        "pst",
        "pdt",
        "akst",
        "akdt",
        "hst",
        "ast",
        "adt",
        "nst",
        "ndt",
        "bst",
        "cet",
        "cest",
        "eet",
        "eest",
        "wet",
        "west",
        "msk",
        "ist",
        "jst",
        "kst",
        "hkt",
        "pkt",
        "wib",
        "wit",
        "wita",
        "aest",
        "aedt",
        "acst",
        "acdt",
        "awst",
        "nzst",
        "nzdt",
        "sast",
        "cat",
        "eat",
        "wat",
        "sst",
        "chst",
        "universal",
        "japan",
        "singapore",
        "hongkong",
        "israel",
        "cuba",
        "egypt",
        "eire",
        "iceland",
        "iran",
        "jamaica",
        "libya",
        "navajo",
        "poland",
        "portugal",
        "turkey",
        "gb",
        "nz",
        "prc",
        "roc",
        "rok",
    ]
)
_TIME_TOKEN = re.compile(
    r"^(?:[\d:./+\-]+(?:[ap]\.?m\.?)?|[a-z]+[+-]?\d[\d:]*[a-z]*\d*|[a-z_]+(?:/[a-z_+\-\d]+)+)$"
)
_RELATIVE_WORD = re.compile(
    r"\b(now|today|tomorrow|yesterday|allballs)\b", re.IGNORECASE
)


def _relative_time_text(text: str) -> bool:
    """Whether the dialect may read `text` as a time relative to the statement."""
    seen = False
    for token in re.split(r"[\s,]+", text.strip().lower()):
        if not token:
            continue
        if token in _RELATIVE_WORDS:
            seen = True
            continue
        # `now+1`: a word with an offset
        word = re.match(r"[a-z]+", token)
        if (
            word is not None
            and word.group() in _RELATIVE_WORDS
            and re.fullmatch(r"[\d:./+\-]+", token[word.end() :])
        ):
            seen = True
        elif token not in _TIME_WORDS and _TIME_TOKEN.match(token) is None:
            return False
    return seen


def _absolute_time_text(text: str) -> bool:
    """A text that is an absolute date or time: it starts with a digit, names no
    `now`.
    """
    stripped = text.strip()
    return (
        stripped[:1].isdigit()
        and _RELATIVE_WORD.search(stripped) is None
        and not _relative_time_text(stripped)
    )


def _is_current_time_literal(node: Any) -> bool:
    """
    A text literal that the database may read as the current time: `'now'::timestamp`,
    SQLite's `datetime('now')`, and wherever its type comes from what it meets (Postgres
    reads `ts - 'now'`, `created < 'now'` and `greatest(ts, 'now')` as times). Only
    where it is clearly text (`s || 'now'`, a cast to text) is it not one.
    """
    if not _relative_time_text(node.name):
        return False
    parent = node.parent
    if isinstance(parent, (exp.Cast, exp.TryCast)):
        return bool(parent.to.is_type(*exp.DataType.TEMPORAL_TYPES))
    return not isinstance(parent, _TEXT_CONTEXTS) and (
        _function_name(parent) not in _TEXT_FUNCTIONS
    )


# Dialects whose text-to-time casts read `'now'` (Postgres' special inputs, Spark's: not
# DuckDB's, whose casts read none, 0.10 to 2.0), and those of them whose catalog says
# which columns hold text (Spark reads `'now'` in a cast of a literal only, so a column
# is proven)
_TEXT_TIME_DIALECTS = frozenset({"postgres", "redshift", "databricks", "spark"})
_TEXT_COLUMN_DIALECTS = frozenset({"postgres", "redshift"})
# Dialects where a user may define a function of any name that is no built-in's
_USER_ROUTINE_DIALECTS = frozenset({"mysql", "mariadb", "clickhouse", "databricks"})
# Typed functions whose value is a date, a time or a number, never text
_NON_TEXT_FUNCTIONS = frozenset(
    {
        "Extract",
        "Year",
        "Month",
        "Day",
        "Hour",
        "Minute",
        "Second",
        "Quarter",
        "Week",
        "WeekOfYear",
        "DayOfWeek",
        "DayOfMonth",
        "DayOfYear",
        "DateAdd",
        "DateSub",
        "DateTrunc",
        "DatetimeTrunc",
        "TimestampTrunc",
        "DateFromParts",
        "TimeFromParts",
        "TimestampFromParts",
        "LastDay",
        "AddMonths",
        "TsOrDsAdd",
        "TsOrDsToDate",
        "TsOrDsToTimestamp",
        "TsOrDsToDatetime",
        "TsOrDsToTime",
        "StrToDate",
        "StrToTime",
        "UnixToTime",
        "TimeStrToTime",
        "TimeStrToDate",
        "Date",
        "Datetime",
        "Time",
        "Timestamp",
        "Abs",
        "Ceil",
        "Floor",
        "Round",
        "Length",
        "DateDiff",
        "TsOrDsDiff",
    }
)


def _unwrapped(node: Any) -> Any:
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _may_be_text(node: Any) -> bool:
    """
    Whether `node` (an operand of a cast to a time) may be a text: a literal that isn't
    an absolute date or time, a concatenation, a function of text, a cast to text, a
    subquery or a CASE. A column is told by the catalog (`temporal_cast_columns`).
    """
    node = _unwrapped(node)
    if isinstance(node, exp.Literal):
        return node.is_string and not _absolute_time_text(node.name)
    if isinstance(node, (exp.Null, exp.Interval, exp.Column, exp.Placeholder)):
        # (a bound parameter's text is judged by its value, `params_time_relative`)
        return False
    if isinstance(node, exp.Parameter):
        return not isinstance(node.this, exp.Literal)
    if isinstance(node, (exp.Cast, exp.TryCast)):
        return not node.to.is_type(
            *exp.DataType.TEMPORAL_TYPES, *exp.DataType.NUMERIC_TYPES
        )
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Neg)):
        return any(_may_be_text(e) for e in node.iter_expressions())
    if isinstance(node, exp.Func) and not isinstance(node, exp.Anonymous):
        return type(node).__name__ not in _NON_TEXT_FUNCTIONS
    return True


# Functions that read their one text argument as a date or time as a cast does
# (Postgres' `date(s)`, `time(s)`, `timestamp(s)`)
_TIME_CONSTRUCTORS = (
    exp.Date,
    exp.Datetime,
    exp.Time,
    exp.Timestamp,
    exp.TsOrDsToDate,
    exp.TsOrDsToDatetime,
    exp.TsOrDsToTime,
    exp.TsOrDsToTimestamp,
    exp.TimeStrToDate,
    exp.TimeStrToTime,
)


def _time_cast_operand(node: Any) -> Any:
    """The operand of a cast to a date or time, or of a function that casts to one."""
    if isinstance(node, (exp.Cast, exp.TryCast)):
        return node.this if node.to.is_type(*exp.DataType.TEMPORAL_TYPES) else None
    if isinstance(node, _TIME_CONSTRUCTORS):
        return node.this
    return None


def _unproven_temporal_cast(node: Any, dialect: str | None) -> bool:
    """
    A cast to a date or time that may read a relative time (`'now'`) from its operand:
    proven only for an operand that isn't text. A column is looked up (Postgres), unless
    the query has relations of its own (a CTE, a derived table, `VALUES`) whose columns
    may be that text; where the catalog can't say, it is refused.
    """
    operand = _unwrapped(_time_cast_operand(node))
    if isinstance(operand, exp.Column):
        if dialect not in _TEXT_COLUMN_DIALECTS:
            # (a column's text, a CTE's or a VALUES' too, isn't read: Databricks)
            return False
        root = node.root()
        return root.find(exp.CTE, exp.Values, exp.Subquery, exp.Lateral) is not None
    return _may_be_text(operand)


def _temporal_cast_columns(stmt: Any, dialect: str | None) -> list[str]:
    """Lower-cased names of the columns cast to a date or time (the catalog says
    text).
    """
    if dialect not in _TEXT_COLUMN_DIALECTS:
        return []
    names: set[str] = set()
    for cast in stmt.find_all(exp.Cast, exp.TryCast, *_TIME_CONSTRUCTORS):
        operand = _time_cast_operand(cast)
        if operand is not None:
            operand = _unwrapped(operand)
            if isinstance(operand, exp.Column):
                names.add(operand.name.lower())
    return sorted(names)


# Window functions whose value depends on the order of rows that tie (`row_number() OVER
# ()`; an order is taken as not unique); rank(), and aggregates over the default frame,
# treat ties alike
_ROW_ORDER_WINDOWS = frozenset(
    {"row_number", "ntile", "first_value", "last_value", "nth_value", "lag", "lead"}
)


# Dialects that take a column that is neither grouped nor aggregated (others refuse it)
_BARE_GROUP_BY_DIALECTS = frozenset({"sqlite", "mysql", "mariadb", "singlestore"})


# Aggregates sqlglot doesn't know (parsed as `Anonymous`): any other unknown function
# may be a scalar one (`printf`), whose bare column is a bare column
_KNOWN_AGGREGATES = frozenset(
    {
        "json_arrayagg",
        "json_objectagg",
        "json_agg",
        "jsonb_agg",
        "group_concat",
        "total",
    }
)


# Aggregates whose float result depends on the order the rows are added in (a parallel
# aggregate adds in another order each run). `Sum` and `Avg` are exact over integers and
# decimals; the others are computed in floats over any argument.
_EXACT_ARGUMENT_AGGREGATES = frozenset({"Sum", "Avg"})
_FLOAT_SENSITIVE_AGGREGATES = frozenset(
    {
        "Sum",
        "Avg",
        "Stddev",
        "StddevPop",
        "StddevSamp",
        "Variance",
        "VariancePop",
        "Corr",
        "CovarPop",
        "CovarSamp",
        "Kurtosis",
        "Skewness",
    }
)
_EXACT_TYPES = frozenset(
    getattr(exp.DataType.Type, name)
    for name in (
        "TINYINT",
        "SMALLINT",
        "INT",
        "BIGINT",
        "INT128",
        "INT256",
        "UTINYINT",
        "USMALLINT",
        "UINT",
        "UBIGINT",
        "UINT128",
        "UINT256",
        "DECIMAL",
        "BIGDECIMAL",
    )
    if hasattr(exp.DataType.Type, name)
)
_DECIMAL_TYPES = frozenset({exp.DataType.Type.DECIMAL, exp.DataType.Type.BIGDECIMAL})


def _exact_operand(node: Any, dialect: str | None) -> bool:
    """
    Whether an aggregate's argument is exact (an integer or decimal, whose sum doesn't
    depend on the order of rows) by what the SQL shows: an integer literal, a cast to an
    integer or decimal type (an integer one on SQLite, whose `DECIMAL` may stay a real),
    a count, and sums and products of those. A column's type isn't shown, so a column is
    not.
    """
    if node is None:
        return True
    if isinstance(node, exp.Distinct):
        return all(_exact_operand(e, dialect) for e in node.expressions)
    if isinstance(node, exp.Literal):
        return not node.is_string and node.name.isdigit()
    if isinstance(node, (exp.Cast, exp.TryCast)):
        to = node.args.get("to")
        return (
            isinstance(to, exp.DataType)
            and to.this in _EXACT_TYPES
            and not (dialect == "sqlite" and to.this in _DECIMAL_TYPES)
        )
    if isinstance(node, (exp.Count, exp.CountIf, exp.Length)):
        return True
    if isinstance(node, (exp.Paren, exp.Neg, exp.Abs)):
        return _exact_operand(node.this, dialect)
    if isinstance(node, (exp.Add, exp.Sub, exp.Mul)):
        return _exact_operand(node.this, dialect) and _exact_operand(
            node.expression, dialect
        )
    return False


def _row_picking_join(node: Any, dialect: str | None) -> bool:
    """
    Whether a join gives other rows (or pairs them otherwise) on another run: `ASOF`
    (which of tied right rows matches), ClickHouse's `ANY` and `SEMI` (any one match,
    with its right columns), and the positional ones (`PASTE`, `POSITIONAL`), which pair
    rows by their place in each side.
    """
    method = str(node.args.get("method") or "").upper()
    kind = str(node.args.get("kind") or "").upper()
    return (
        method in ("ASOF", "POSITIONAL")
        or kind in ("ASOF", "PASTE", "ANY")
        or (dialect == "clickhouse" and kind == "SEMI")
    )


def _inexact_aggregate(node: Any, dialect: str | None) -> bool:
    """
    Whether `node` is a float-sensitive aggregate of an argument not shown exact (or
    SQLite's `total`, a float sum).
    """
    if isinstance(node, exp.Anonymous):
        return dialect == "sqlite" and _function_name(node) == "total"
    name = type(node).__name__
    return name in _FLOAT_SENSITIVE_AGGREGATES and not (
        name in _EXACT_ARGUMENT_AGGREGATES
        and all(
            _exact_operand(a, dialect)
            for a in (node.args.get("this"), node.args.get("expression"))
        )
    )


# ClickHouse settings that change how a query runs, not what it returns (those with an
# `*_overflow_mode` of `any` or `break`, `max_rows_to_*` limits and the like cut or
# change the result)
_HARMLESS_SETTINGS = frozenset(
    {
        "max_threads",
        "max_block_size",
        "max_execution_time",
        "max_memory_usage",
        "max_bytes_before_external_group_by",
        "max_bytes_before_external_sort",
        "log_comment",
    }
)


def _unsafe_settings(node: Any) -> bool:
    """Whether a query's `SETTINGS` clause has a setting outside the harmless ones."""
    settings = node.args.get("settings")
    if not settings:
        return False
    return not all(
        isinstance(s, exp.EQ)
        and isinstance(s.this, (exp.Column, exp.Identifier, exp.Var))
        and s.this.name.lower() in _HARMLESS_SETTINGS
        for s in settings
    )


def _any_unsafe_settings(stmt: Any) -> bool:
    return any(_unsafe_settings(q) for q in stmt.find_all(exp.Query))


def _positional_join(stmt: Any) -> bool:
    """Whether a join pairs rows by their place (`PASTE`, `POSITIONAL`)."""
    return any(
        str(j.args.get("method") or "").upper() == "POSITIONAL"
        or str(j.args.get("kind") or "").upper() == "PASTE"
        for j in stmt.find_all(exp.Join)
    )


def _is_scalar_min_max(node: Any) -> bool:
    """SQLite's `max(a, b)` / `min(a, b)` (sqlglot parses them as the aggregates)."""
    return isinstance(node, (exp.Max, exp.Min)) and bool(node.args.get("expressions"))


def _is_aggregate_query(select: Any) -> bool:
    """
    Whether a select without `GROUP BY` is an aggregate query (one group): an aggregate
    in its own select list or `HAVING`, not in a window, a subquery or a scalar
    function.
    """
    stack = [*select.expressions]
    if (having := select.args.get("having")) is not None:
        stack.append(having)
    while stack:
        node = stack.pop()
        if isinstance(node, (exp.Window, exp.Subquery, exp.Select)) or (
            _is_scalar_min_max(node)
        ):
            if _is_scalar_min_max(node):
                stack.extend(node.iter_expressions())
            continue
        if isinstance(node, exp.AggFunc) or (
            isinstance(node, exp.Anonymous) and node.name.lower() in _KNOWN_AGGREGATES
        ):
            return True
        stack.extend(node.iter_expressions())
    return False


def _qualified_key(column: Any) -> str:
    """
    A qualified column as `catalog.schema.table.name`, each part folded to lower case
    unless quoted (`s1.t.id` is not `s2.t.id`, nor `"T".id` `"t".id`).
    """
    parts = []
    for key in ("catalog", "db", "table", "this"):
        part = column.args.get(key)
        if part is not None:
            parts.append(
                part.name if getattr(part, "quoted", False) else part.name.lower()
            )
    return ".".join(parts)


def _bare_column_in_group_by(select: Any) -> bool:
    """
    Whether a `GROUP BY` query (or an aggregate query without one) selects (or filters
    with `HAVING`) a column that is neither grouped nor inside an aggregate: its value
    comes from any row of the group.
    """
    group = select.args.get("group")
    if group is None and not _is_aggregate_query(select):
        return False
    projections = select.expressions
    grouped: set[str] = set()
    # grouped columns by name, qualified ones (`a.id`) also by table: `b.id` isn't
    # grouped by `GROUP BY a.id`
    grouped_names: set[str] = set()
    grouped_unqualified: set[str] = set()
    if group is not None and group.args.get("all"):
        return False
    by_alias = {
        p.alias.lower(): p.this for p in projections if isinstance(p, exp.Alias)
    }
    for node in group.walk() if group is not None else ():
        if isinstance(node, exp.Column) and not node.table:
            # `GROUP BY l` for `lower(g) AS l`: the expression it names
            if (named := by_alias.get(node.name.lower())) is not None:
                grouped.add(named.sql().lower())
                if isinstance(named, exp.Column):
                    # `GROUP BY k` for `g AS k` (or `t.g AS k`) groups `g`
                    grouped_names.add(named.name.lower())
                    if named.table:
                        grouped.add(_qualified_key(named))
                    else:
                        grouped_unqualified.add(named.name.lower())
        if isinstance(node, exp.Literal) and node.is_int:
            # `GROUP BY 1`: the projection it numbers
            index = int(node.name) - 1
            if 0 <= index < len(projections):
                node = projections[index]
                node = node.this if isinstance(node, exp.Alias) else node
        if isinstance(node, exp.Column):
            grouped_names.add(node.name.lower())
            if node.table:
                grouped.add(_qualified_key(node))
            else:
                grouped_unqualified.add(node.name.lower())
        elif isinstance(node, exp.Var) and node.name.upper() == "ALL":
            return False
        elif not isinstance(node, (exp.Group, exp.Tuple, exp.Rollup, exp.Cube)):
            grouped.add(node.sql().lower())
    aliases = {p.alias.lower() for p in projections if isinstance(p, exp.Alias)}

    def column_is_bare(node: Any, is_having: bool) -> bool:
        name = node.name.lower()
        if node.table:
            is_grouped = _qualified_key(node) in grouped or name in grouped_unqualified
        else:
            is_grouped = name in grouped_names
        return not is_grouped and not (is_having and name in aliases)

    def bare(node: Any, is_having: bool) -> bool:
        if isinstance(node, (exp.Subquery, exp.Select)):
            # a correlated column (one the subquery's own tables don't qualify) is the
            # outer query's, whatever the subquery aggregates; an unqualified one may be
            # (an aliased table is named by its alias only: `t AS i` hides `t`)
            own = {
                t.alias_or_name.lower()
                for t in node.find_all(exp.Table, exp.Subquery, exp.CTE)
            } - {""}
            return any(
                column_is_bare(c, is_having)
                for c in node.find_all(exp.Column)
                if not c.is_star and not (c.table and c.table.lower() in own)
            )
        if isinstance(node, exp.Window):
            # `sum(x) OVER ()` runs over the grouped rows: `x` is a bare column unless
            # grouped or aggregated inside (`sum(sum(x)) OVER ()`)
            return any(
                bare(child, is_having) for child in node.this.iter_expressions()
            ) or any(
                bare(child, is_having)
                for child in node.iter_expressions()
                if child is not node.this
            )
        if (
            not _is_scalar_min_max(node)
            and isinstance(
                node,
                (
                    exp.AggFunc,
                    # `count(*) FILTER (WHERE x > 0)`, `... WITHIN GROUP (ORDER BY x)`
                    exp.Filter,
                    exp.WithinGroup,
                ),
            )
        ) or (
            isinstance(node, exp.Anonymous) and node.name.lower() in _KNOWN_AGGREGATES
        ):
            return False
        if isinstance(node, exp.Column):
            return column_is_bare(node, is_having)
        if not isinstance(node, exp.Star) and node.sql().lower() in grouped:
            return False
        return any(bare(child, is_having) for child in node.iter_expressions())

    if any(
        isinstance(p, exp.Star) or (isinstance(p, exp.Column) and p.is_star)
        for p in projections
    ):
        return True
    having = select.args.get("having")
    return any(
        bare(p.this if isinstance(p, exp.Alias) else p, False) for p in projections
    ) or (having is not None and bare(having.this, True))


def _picks_arbitrary_rows(stmt: Any, dialect: str | None) -> bool:
    """
    Whether the query may give other rows, or number them otherwise, each time it runs:
    a window function that tells tied rows apart (or has a `ROWS` / `GROUPS` frame),
    `DISTINCT ON`, a bare column next to a `GROUP BY` (SQLite, and MySQL without
    `ONLY_FULL_GROUP_BY`, take it from any row of the group), or Oracle's `ROWNUM`.
    Aggregates that pick any value or depend on the order of rows aren't in the
    allowlist (`_PROVEN_FUNCTIONS`).
    """
    if dialect in _BARE_GROUP_BY_DIALECTS and any(
        _bare_column_in_group_by(select) for select in stmt.find_all(exp.Select)
    ):
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
    if dialect == "sqlite" and any(
        _multi_row_scalar_subquery(q) for q in stmt.find_all(exp.Subquery)
    ):
        return True
    return any(d.args.get("on") is not None for d in stmt.find_all(exp.Distinct))


def _multi_row_scalar_subquery(node: Any) -> bool:
    """
    Whether a subquery is a scalar one that may return several rows, of which SQLite
    takes the first it scans. One aggregate row (no `GROUP BY`) or a `SELECT` without
    `FROM` is one row.
    """
    if isinstance(
        node.parent, (exp.From, exp.Join, exp.SetOperation, exp.Subquery)
    ) or (isinstance(node.parent, exp.In) and node.arg_key == "query"):
        return False
    query = node.this
    if not isinstance(query, exp.Select):
        return True
    if query.args.get("from_", query.args.get("from")) is None:
        return False

    def own(select: Any, kind: Any) -> bool:
        # (one of its own, not of a query nested in it)
        return any(n.find_ancestor(exp.Select) is query for n in select.find_all(kind))

    return (
        query.args.get("group") is not None
        or any(own(s, exp.Window) for s in query.selects)
        or not all(own(s, exp.AggFunc) for s in query.selects)
    )


def _collapses_rows(stmt: Any) -> bool:
    """
    Whether the query merges rows that compare equal into one (`GROUP BY`, `SELECT
    DISTINCT`, `UNION`, `INTERSECT`, `EXCEPT`, `DISTINCT` in an aggregate other than
    `COUNT`): under a collation that equates different texts, the database returns
    any one of them.
    """
    if stmt.find(exp.Group, exp.Min, exp.Max) is not None:
        return True
    for d in stmt.find_all(exp.Distinct):
        if not isinstance(d.parent, exp.Count):
            return True
    return any(
        not isinstance(op, exp.Union) or op.args.get("distinct", True)
        for op in stmt.find_all(exp.SetOperation)
    )


def _binary_collation(name: str, dialect: str | None) -> bool:
    """
    Whether collation `name` (lower-cased) equates only identical texts, where the
    dialect pads nothing: MySQL's `utf8mb4_bin` pads (`'a'` is `'a '`),
    `utf8mb4_0900_bin`
    and MariaDB's `utf8mb4_nopad_bin` don't; SQL Server pads in every collation;
    a user's Postgres collation may be nondeterministic whatever its name (the catalog
    says: a `COLLATE` of it is refused here).
    """
    if dialect in ("postgres", "redshift"):
        return name in ("c", "posix", "ucs_basic")
    if dialect in ("mysql", "mariadb"):
        return name == "binary" or name.endswith(("_0900_bin", "_nopad_bin"))
    if dialect == "tsql":
        return False
    return name in ("utf8_binary", "binary", "c", "posix", "") or name.endswith(
        ("_bin", "_bin2")
    )


def _uses_collation(tokens: list[Any], dialect: str | None = None) -> bool:
    """
    Whether the query names a collation other than a binary one (`COLLATE x`,
    `collate(s, 'x')`). A column's own collation is the catalog's to say.
    """
    for i, t in enumerate(tokens):
        if t.text.lower() != "collate":
            continue
        nxt = tokens[i + 1].text.lower().strip("'\"`") if i + 1 < len(tokens) else ""
        if _binary_collation(nxt, dialect) and nxt:
            continue
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


def _outer_order_by(
    stmt: Any,
    dialect: str | None,
    star_columns: list[str] | None = None,
    star_ordinals: bool = True,
) -> str | None:
    """
    The query's top-level `ORDER BY` (without the keyword) on its output columns,
    for a wrapper that repeats it outside; `None` if there is none, or a key is not an
    output column (a selected column, its alias, an expression selected as it is, or its
    position) of an unambiguous select list. A select list of a lone `*` (or `t.*`) is
    the planned columns `star_columns`, where given; any other star is not expanded. A
    position in it needs `star_ordinals`: the columns are the database's own, in its
    order
    (not a user's `schema=`).
    """
    while not stmt.args.get("order"):
        if not isinstance(stmt, (exp.Subquery, exp.Paren)):
            return None
        stmt = stmt.this
    selects = stmt.selects
    if not selects:
        return None
    expanded = False
    if any(isinstance(s, exp.Star) or s.is_star for s in selects):
        if (
            star_columns is None
            or len(selects) != 1
            or isinstance(stmt, exp.SetOperation)
        ):
            return None
        selects = [exp.column(c, quoted=True) for c in star_columns]
        expanded = True
        if not selects:
            return None
    names = [s.alias_or_name for s in selects]
    # (the servers that need this compare column names without regard to case)
    lowered = [n.lower() for n in names]
    if not all(names) or len(set(lowered)) != len(names):
        return None
    is_set_operation = isinstance(stmt, exp.SetOperation)
    keys = []
    for o in stmt.args["order"].expressions:
        if not isinstance(o, exp.Ordered):
            return None
        key = o.this
        while isinstance(key, exp.Paren):
            key = key.this
        name = None
        if isinstance(key, exp.Literal) and not key.is_string:
            try:
                ordinal = int(key.name)
            except ValueError:
                return None
            if (star_ordinals or not expanded) and 1 <= ordinal <= len(names):
                name = names[ordinal - 1]
        elif (
            isinstance(key, exp.Column)
            # (`t.id` of `SELECT * FROM t`: the one table's column)
            and (not key.table or (expanded and not stmt.args.get("joins")))
            and key.name.lower() in lowered
        ):
            # an alias, or the selected column of that name
            name = names[lowered.index(key.name.lower())]
        if name is None and not is_set_operation:
            for s, n in zip(selects, names, strict=True):
                if key == (s.this if isinstance(s, exp.Alias) else s):
                    name = n
                    break
        if name is None:
            return None
        ordered = o.copy()
        # qualified by the wrapper's derived table (`_pl`, `relation_qualifier`): the
        # inner column, not the select item that converts it (a decimal read as text
        # would sort as text)
        ordered.set(
            "this",
            exp.Column(
                this=exp.to_identifier(name, quoted=True),
                table=exp.to_identifier("_pl", quoted=True),
            ),
        )
        keys.append(_escaped_identifiers(ordered, dialect).sql(dialect=dialect))
    return ", ".join(keys)


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


def _tables(stmt: Any, dialect: str | None) -> tuple[list[list[str]], list[list[str]]]:
    """
    The tables (or views) a query reads, as `[catalog.][schema.]name` parts, and those
    of
    them read only with `FINAL` (ClickHouse), unquoted
    names folded as the dialect folds them (`Orders` is `orders` on Postgres, `ORDERS`
    on Oracle): not its CTEs. A table function is none (it isn't proven, `_unproven`).
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
    plain: list[list[str]] = []
    for t in stmt.find_all(exp.Table):
        if isinstance(t.this, exp.Identifier):
            name = t.name
        else:
            continue
        if not name or _cte_in_scope(t, dialect):
            continue
        parts = [p for p in (t.catalog, t.db, name) if p]
        if parts not in tables:
            tables.append(parts)
        if not isinstance(t.parent, exp.Final):
            plain.append(parts)
    return tables, [t for t in tables if t not in plain]


# Where the user can replace a built-in function (SQLite's `create_function`, a DuckDB
# macro or Python function, a Postgres or Oracle function in a schema the name is looked
# up in before the built-ins), so that the name sqlglot knows runs something else
_OVERRIDABLE_DIALECTS = frozenset(
    {
        "sqlite",
        "duckdb",
        "postgres",
        "oracle",
        "mysql",
        "mariadb",
        "clickhouse",
        "databricks",
    }
)


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
    if stmt is not None and dialect in ("postgres", "redshift"):
        # attribute notation calls a function of the row: `c.f` and `(c).f` are `f(c)`
        # where `c` has no column `f`
        names.update(
            c.name.lower()
            for c in stmt.find_all(exp.Column)
            if c.table and c.name and not c.is_star
        )
        names.update(
            d.expression.name.lower()
            for d in stmt.find_all(exp.Dot)
            if isinstance(d.expression, exp.Identifier)
        )
    if stmt is not None and dialect == "oracle":
        # a function without parentheses is read as a column where no column has its
        # name
        names.update(
            c.name.lower()
            for c in stmt.find_all(exp.Column)
            if not c.table and c.name and not c.is_star
        )
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


def _non_builtin_calls(stmt: Any, tokens: list[Any], dialect: str | None) -> list[str]:
    """
    Lower-cased names the SQL calls (as written: sqlglot reads `nvl(x)` as `COALESCE`,
    `strpos` as its own `STR_POSITION`) that are no built-in function of the dialect
    (`BUILTIN_FUNCTIONS`), where a user may create one of any name (MySQL, MariaDB,
    ClickHouse, Databricks): a stored, loadable, SQL or temporary function, which the
    catalog may not show. Not a type's name after `AS` or `::` (`AS DECIMAL(10, 2)`) or
    an
    alias's column list (`AS t (a, b)`, `WITH c (a, b)`).
    """
    builtins = BUILTIN_FUNCTIONS.get(dialect or "")
    if builtins is None:
        return []
    aliases = {a.name.lower() for a in stmt.find_all(exp.TableAlias)}
    kind = sqlglot.TokenType
    calls: set[str] = set()
    i, n = 0, len(tokens)
    while i < n:
        t = tokens[i]
        if t.text.lower() == "as" or t.token_type == kind.DCOLON:
            # a type or an alias: its name and its parenthesised arguments are no call
            j = i + 1
            if j < n and tokens[j].text.isidentifier():
                j += 1
                if j < n and tokens[j].token_type == kind.L_PAREN:
                    depth = 0
                    while j < n:
                        if tokens[j].token_type == kind.L_PAREN:
                            depth += 1
                        elif tokens[j].token_type == kind.R_PAREN:
                            depth -= 1
                            if depth == 0:
                                j += 1
                                break
                        j += 1
                i = j
                continue
        nxt = tokens[i + 1] if i + 1 < n else None
        word = t.text.lower()
        if (
            nxt is not None
            and nxt.token_type == kind.L_PAREN
            and t.text.isidentifier()
            and "STRING" not in t.token_type.name
            and word not in SYNTAX_WORDS
            and word not in aliases
            and word not in builtins
        ):
            calls.add(word)
        i += 1
    return sorted(calls)


def _cast_type_names(stmt: Any, dialect: str | None) -> list[str]:
    """Lower-cased type names the query writes (a user's cast is found by them)."""
    names: set[str] = set()
    for dt in stmt.find_all(exp.DataType):
        with contextlib.suppress(sqlglot.errors.SqlglotError):
            names.add(dt.sql(dialect=dialect).split("(")[0].strip().lower())
    if dialect in ("postgres", "redshift"):
        # the types of its literals (marked `~`: they stand for operands, not for the
        # target of a cast), for the operators a user defined on them (`0.5 * 2` runs a
        # user's `*`(numeric, int4)), which no column or cast of the query names
        literal: set[str] = set()
        for lit in stmt.find_all(exp.Literal):
            if lit.is_string:
                literal.update(("text", "unknown"))
            elif lit.name.isdigit():
                literal.update(("int4", "int8"))
            else:
                literal.add("numeric")
        if stmt.find(exp.Boolean) is not None:
            literal.add("bool")
        names.update(f"~{n}" for n in literal)
    return sorted(n for n in names if n)


def _referenced_columns(stmt: Any, dialect: str | None = None) -> list[str] | None:
    """
    Lower-cased names the query writes, `None` where a `*` projects every column (or,
    on Postgres, a table's name stands for its whole row: `SELECT DISTINCT t FROM t`).
    """
    for select in stmt.find_all(exp.Select):
        if any(e.is_star for e in select.expressions):
            return None
    if dialect in ("postgres", "redshift"):
        relations = _relations(stmt)
        if any(
            not c.table and c.name.lower() in relations
            for c in stmt.find_all(exp.Column)
        ):
            return None
    return sorted({i.name.lower() for i in stmt.find_all(exp.Identifier)})


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
        "outer_order_by": None,
        "output_columns": None,
        "read_only": False,
        "repeatable": False,
        "top_level_limit": False,
        "with_ties": False,
        "nested_queries": False,
        "side_effects": False,
        "side_effects_outside_rows": False,
        "volatile_columns": [],
        "all_columns_volatile": False,
        "known_functions": [],
        "unproven": [],
        "tables": [],
        "collapses_rows": False,
        "collated": False,
        "cast_types": [],
        "any_operator": False,
        "positional_join": False,
        "unsafe_settings": False,
        "referenced_columns": None,
        "final_tables": [],
        "temporal_cast_columns": [],
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
    # (its tables are unknown: a user's operator, which sqlglot doesn't parse either,
    # may be on any type)
    facts["any_operator"] = dialect in ("postgres", "redshift")
    return facts


def outer_order_by(
    sql: str,
    dialect: str | None,
    columns: list[str],
    star_ordinals: bool = True,
) -> str | None:
    """
    The query's `ORDER BY` on its output columns (see `_outer_order_by`), where a
    `SELECT *` has the planned `columns`; `None` if it can't be kept that way.
    """
    try:
        statements, _ = _parse_query(_without_lambda_parentheses(sql, dialect), dialect)
    except sqlglot.errors.SqlglotError:
        return None
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return None
    return _outer_order_by(statements[0], dialect, columns, star_ordinals)


# SQL Server's reserved keywords ("Reserved Keywords (Transact-SQL)"). It rejects one as
# an unquoted alias, so a query that has one as an alias isn't valid to it as
# written: but it runs statements with no `;` between them, and sqlglot parses
# `SELECT … FROM c EXEC('…')` as `c`'s alias `EXEC`. Such a query isn't read only.
_TSQL_RESERVED = frozenset(
    {
        "ADD", "ALL", "ALTER", "AND", "ANY", "AS", "ASC", "AUTHORIZATION", "BACKUP",
        "BEGIN", "BETWEEN", "BREAK", "BROWSE", "BULK", "BY", "CASCADE", "CASE", "CHECK",
        "CHECKPOINT", "CLOSE", "CLUSTERED", "COALESCE", "COLLATE", "COLUMN", "COMMIT",
        "COMPUTE", "CONSTRAINT", "CONTAINS", "CONTAINSTABLE", "CONTINUE", "CONVERT",
        "CREATE", "CROSS", "CURRENT", "CURRENT_DATE", "CURRENT_TIME",
        "CURRENT_TIMESTAMP", "CURRENT_USER", "CURSOR", "DATABASE", "DBCC", "DEALLOCATE",
        "DECLARE", "DEFAULT", "DELETE", "DENY", "DESC", "DISK", "DISTINCT",
        "DISTRIBUTED", "DOUBLE", "DROP", "DUMP", "ELSE", "END", "ERRLVL", "ESCAPE",
        "EXCEPT", "EXEC", "EXECUTE", "EXISTS", "EXIT", "EXTERNAL", "FETCH", "FILE",
        "FILLFACTOR", "FOR", "FOREIGN", "FREETEXT", "FREETEXTTABLE", "FROM", "FULL",
        "FUNCTION", "GOTO", "GRANT", "GROUP", "HAVING", "HOLDLOCK", "IDENTITY",
        "IDENTITY_INSERT", "IDENTITYCOL", "IF", "IN", "INDEX", "INNER", "INSERT",
        "INTERSECT", "INTO", "IS", "JOIN", "KEY", "KILL", "LEFT", "LIKE", "LINENO",
        "LOAD", "MERGE", "NATIONAL", "NOCHECK", "NONCLUSTERED", "NOT", "NULL", "NULLIF",
        "OF", "OFF", "OFFSETS", "ON", "OPEN", "OPENDATASOURCE", "OPENQUERY",
        "OPENROWSET", "OPENXML", "OPTION", "OR", "ORDER", "OUTER", "OVER", "PERCENT",
        "PIVOT", "PLAN", "PRECISION", "PRIMARY", "PRINT", "PROC", "PROCEDURE", "PUBLIC",
        "RAISERROR", "READ", "READTEXT", "RECONFIGURE", "REFERENCES", "REPLICATION",
        "RESTORE", "RESTRICT", "RETURN", "REVERT", "REVOKE", "RIGHT", "ROLLBACK",
        "ROWCOUNT", "ROWGUIDCOL", "RULE", "SAVE", "SCHEMA", "SECURITYAUDIT", "SELECT",
        "SEMANTICKEYPHRASETABLE", "SEMANTICSIMILARITYDETAILSTABLE",
        "SEMANTICSIMILARITYTABLE", "SESSION_USER", "SET", "SETUSER", "SHUTDOWN", "SOME",
        "STATISTICS", "SYSTEM_USER", "TABLE", "TABLESAMPLE", "TEXTSIZE", "THEN", "TO",
        "TOP", "TRAN", "TRANSACTION", "TRIGGER", "TRUNCATE", "TRY_CONVERT", "TSEQUAL",
        "UNION", "UNIQUE", "UNPIVOT", "UPDATE", "UPDATETEXT", "USE", "USER", "VALUES",
        "VARYING", "VIEW", "WAITFOR", "WHEN", "WHERE", "WHILE", "WITH", "WITHIN",
        "WRITETEXT",
    }
)  # fmt: skip


def _hides_statement(
    statements: list[Any], tokens: list[Any], dialect: str | None
) -> bool:
    """
    Whether an alias may be another statement's start (SQL Server only).

    An unquoted alias, bare or after `AS`, that is a reserved keyword (`c EXEC(…)`,
    `c COMMIT`), or a bare alias with a column list after a table name (`c EXEC([…])` is
    `c` with the alias `EXEC` and the columns `[…]`; SQL Server has no such table alias,
    where a derived table, `VALUES` or `APPLY` has `v(id, s)`). Fail closed: such a
    query isn't read only, whatever the AST says.
    """
    if dialect != "tsql":
        return False
    by_start = {t.start: i for i, t in enumerate(tokens)}

    def bare(ident: Any) -> bool:
        i = by_start.get(ident.meta.get("start", -1)) if ident.meta else None
        # (no position, or the token isn't the one the identifier was made of: unknown)
        if i is None:
            return True
        return i == 0 or tokens[i - 1].token_type != sqlglot.TokenType.ALIAS

    for stmt in statements:
        for node in stmt.find_all(exp.TableAlias, exp.Alias):
            alias = node.args.get("alias") if isinstance(node, exp.Alias) else node.this
            names = [alias]
            if isinstance(node, exp.TableAlias):
                names += node.args.get("columns") or []
            for ident in names:
                if (
                    isinstance(ident, exp.Identifier)
                    and not ident.args.get("quoted")
                    and ident.name.upper() in _TSQL_RESERVED
                ):
                    return True
            if (
                isinstance(alias, exp.Identifier)
                and isinstance(node, exp.TableAlias)
                and node.args.get("columns")
                # (a derived table, `VALUES` or `APPLY` takes a bare alias and columns)
                and isinstance(node.parent, exp.Table)
                and isinstance(node.parent.this, exp.Identifier)
                and bare(alias)
            ):
                return True
    return False


def _filtered_aliases(stmt: Any) -> set[str]:
    """
    The lower-cased names a query's `HAVING` or `QUALIFY` uses, and those of the output
    aliases they name in turn (`net > 0 AS profitable … HAVING profitable`).
    """

    def aliases(node: Any) -> list[str]:
        # (a name inside an aggregate of this query is an input column's; one in a
        # subquery may be an outer alias)
        return [
            c.name.lower()
            for c in node.find_all(exp.Column)
            if not c.table
            and not (
                (agg := c.find_ancestor(exp.AggFunc)) is not None
                and agg.find_ancestor(exp.Select) is stmt
            )
        ]

    names = {
        n
        for clause in (stmt.args.get("having"), stmt.args.get("qualify"))
        if clause is not None
        for n in aliases(clause)
    }
    selects = {s.alias_or_name.lower(): s for s in stmt.selects}
    pending = list(names)
    while pending:
        select = selects.get(pending.pop())
        if select is None:
            continue
        for name in aliases(select):
            if name not in names:
                names.add(name)
                pending.append(name)
    return names


def _executable_comment(tokens: list[Any], dialect: str | None) -> bool:
    """
    Whether a MySQL executable comment (`/*! … */`, MariaDB's `/*M! … */`) is in the
    query: the server runs its text as SQL, which the parser took as a comment.
    """
    return dialect in ("mysql", "mariadb", "singlestore") and any(
        c.startswith(("!", "M!")) for t in tokens for c in t.comments
    )


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
    executable = _executable_comment(tokens, dialect)
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
    facts["read_only"] = (
        len(statements) == 1
        and _is_read_only(statements[0])
        and not _hides_statement(statements, tokens, dialect)
    )
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return facts

    stmt = statements[0]
    facts["single_select"] = True
    facts["has_cte"] = stmt.args.get("with_") is not None or (
        stmt.args.get("with") is not None
    )
    facts["top_level_order_by_without_limit"] = _top_level_order_by_without_limit(stmt)
    facts["outer_order_by"] = _outer_order_by(stmt, dialect)
    # what can't take another `LIMIT` after it
    facts["top_level_limit"] = any(
        stmt.args.get(k) for k in ("limit", "fetch", "offset", "locks")
    )
    # `FETCH FIRST n ROWS WITH TIES`, `TOP n WITH TIES`: a LIMIT around it may be pushed
    # into it by the server (Trino), past the ties
    facts["with_ties"] = any(
        isinstance(node, exp.LimitOptions) and node.args.get("with_ties")
        for node in stmt.walk()
    )
    facts["nested_queries"] = _nested_queries(stmt)
    relations = _relations(stmt)
    # what the allowlist doesn't prove (`_unproven`): any function, operator, column or
    # table of the query
    unproven = [node for node in stmt.walk() if _unproven(node, dialect, relations)]
    # ... and the names written that are no built-in's: judged as written, the nodes
    # they parse to flagged (a name no node is found for may be any column's)
    written = _non_builtin_calls(stmt, tokens, dialect)
    flagged: list[Any] = []
    matched: set[str] = set()
    if written:
        for f in stmt.find_all(exp.Func):
            names = (
                {_function_name(f)}
                if isinstance(f, exp.Anonymous)
                else {n.lower() for n in f.sql_names()}
            )
            if names & set(written):
                matched |= names & set(written)
                flagged.append(f)
        unproven.extend(f for f in flagged if not any(f is u for u in unproven))
    # A float aggregate's value changes in its last bits between runs (a parallel
    # aggregate adds in another order): as an output column it isn't a partition key
    # (the column is listed as volatile), anywhere else (a `HAVING`, a subquery, a join
    # condition) it decides which rows there are.
    projections = (
        {id(p) for p in stmt.selects} if isinstance(stmt, exp.Select) else set()
    )
    # (the names a `HAVING` or `QUALIFY` uses: an output alias there is its expression,
    # which may name other aliases)
    filtered = _filtered_aliases(stmt)
    inexact: list[Any] = []
    for node in stmt.find_all(exp.Func):
        if not _inexact_aggregate(node, dialect):
            continue
        top = node
        while top is not None and id(top) not in projections and top is not stmt:
            top = top.parent
        if top is not None and id(top) in projections:
            if (
                node.find_ancestor(exp.Select) is stmt
                and top.alias_or_name.lower() not in filtered
            ):
                inexact.append(node)
                continue
        if not any(node is u for u in unproven):
            unproven.append(node)
    columns = _volatile_columns(stmt, [*unproven, *inexact])
    if set(written) - matched:
        columns = None
    if columns is None or any(
        _function_name(f) in _ROWS_READ_FUNCTIONS for f in stmt.find_all(exp.Func)
    ):
        facts["all_columns_volatile"] = True
    else:
        facts["volatile_columns"] = columns
    facts["known_functions"] = _known_functions(stmt, tokens, dialect)
    facts["collapses_rows"] = _collapses_rows(stmt)
    facts["positional_join"] = _positional_join(stmt)
    facts["unsafe_settings"] = _any_unsafe_settings(stmt)
    facts["collated"] = _uses_collation(tokens, dialect)
    facts["unproven"] = sorted(
        {_describe(n, dialect) for n in unproven if not any(n is f for f in flagged)}
        | {f"{n}()" for n in written}
        | ({"an executable comment (/*! */)"} if executable else set())
    )[:5]
    # A view among them may call a volatile function (`catalog::view_definition`).
    facts["tables"], facts["final_tables"] = _tables(stmt, dialect)
    facts["referenced_columns"] = _referenced_columns(stmt, dialect)
    facts["temporal_cast_columns"] = _temporal_cast_columns(stmt, dialect)
    if dialect in ("postgres", "redshift"):
        facts["cast_types"] = _cast_type_names(stmt, dialect)
    # anywhere in the query: a nested LIMIT without a unique ORDER BY picks rows too
    facts["repeatable"] = (
        stmt.find(exp.Limit, exp.Fetch, exp.Offset, exp.TableSample) is None
        and not unproven
        and not written
        and not _picks_arbitrary_rows(stmt, dialect)
        and not executable
    )
    # Running it may change something: a function sqlglot doesn't know (`nextval`, or a
    # user's, which may write), a sequence's pseudo-column, a variable assignment.
    facts["side_effects"] = bool(written) or any(
        not isinstance(n, exp.Func) or isinstance(n, exp.Anonymous) for n in unproven
    )
    # Where MariaDB evaluates such a function under a `LIMIT 0`: a set operation's
    # branches, a query without FROM, and conditions (`WHERE f() = 1` is constant)
    calls = [
        node
        for node in unproven
        if not isinstance(node, exp.Func)
        or isinstance(node, exp.Anonymous)
        or any(node is f for f in flagged)
    ]
    facts["side_effects_outside_rows"] = (
        bool(calls) or bool(set(written) - matched)
    ) and (
        isinstance(stmt, exp.SetOperation)
        or stmt.args.get("from_", stmt.args.get("from")) is None
        or any(c.find_ancestor(exp.Where, exp.Having, exp.Join) for c in calls)
        or bool(set(written) - matched)
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
    # (its own aliases, which a `HAVING` may name)
    renamed: str | None = None
    if (
        names is not None
        and isinstance(stmt, exp.Select)
        and len(stmt.selects) == len(names)
        and not any(s.is_star for s in stmt.selects)
    ):
        with contextlib.suppress(sqlglot.errors.SqlglotError):
            renamed = stmt.sql(dialect=dialect)
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
    if renamed is not None and not inspect_query(renamed, dialect)["repeatable"]:
        facts["repeatable"] = False
    with contextlib.suppress(sqlglot.errors.SqlglotError):
        tokens = sqlglot.Dialect.get_or_raise(dialect).tokenize(sql)
        written = _known_functions(None, tokens, dialect)
        facts["known_functions"] += [
            n for n in written if n not in facts["known_functions"]
        ]
    if not named:
        # its names aren't the query's: a volatile column may be any of them
        facts["output_columns"] = names
        if facts["volatile_columns"]:
            facts["volatile_columns"] = []
            facts["all_columns_volatile"] = True
    return facts
