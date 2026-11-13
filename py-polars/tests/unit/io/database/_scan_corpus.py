"""
Shared data for the `scan_database` differential tests.

`edge_rows()` is the edge-case table (every column has NULLs), and `PREDICATES` is the
fixed predicate corpus. The SQLite and server test files, and the scripts that seed
the test servers, all use them, so SQLite runs the same corpus on every PR.

Most cells of a backend matrix plan the same SQL for a predicate: the dialect and the
schema decide it, not the driver. By default one base cell per backend runs the whole
corpus on the table, and everything else (other cells, the query modes) runs the
predicates it plans differently, plus a sample (`cell_predicates`).
`POLARS_TEST_DATABASE_FULL=1` (the nightly job) runs the whole corpus on every cell.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    from polars._typing import PolarsDataType

# Logical columns of the edge table, in order. Dialects map them to their own types.
EDGE_COLUMNS = [
    "id",
    "i16",
    "i32",
    "i64",
    "f64",
    "dec",
    "s",
    "sb",
    "d",
    "ts",
    "b",
]

I16 = [-32768, 32767, 0, -1, 7]
I32 = [-(2**31), 2**31 - 1, 0, 42, -42]
I64 = [-(2**63), 2**63 - 1, 0, 10**12, -5]
F64 = [0.0, -0.0, 1.5, -1.5, float("nan"), float("inf"), float("-inf"), 1e300, 2.5]
DEC = [
    Decimal("0"),
    Decimal("12.5"),
    Decimal("-0.0001"),
    Decimal("99999999999999.9999"),
]
STRINGS = [
    "abc",
    "ABC",
    "Abc",
    "abc ",
    "  abc",
    "",
    "%",
    "_",
    "\\",
    "'",
    "a%b",
    "a_b",
    "é",
    "e",
    "É",
    "😀",
    "ß",
    "ss",
    "b",
    "B",
]
DATES = [
    date(1900, 1, 1),
    date(1969, 12, 31),
    date(1970, 1, 1),
    date(2026, 3, 29),
    date(9999, 12, 31),
]
TIMESTAMPS = [
    datetime(1900, 1, 1),
    datetime(1970, 1, 1, 0, 0, 0, 1),
    datetime(2026, 3, 29, 1, 30),
    datetime(2026, 10, 25, 2, 30, 0, 500000),
    datetime(9999, 12, 31, 23, 59, 59, 999999),
]
BOOLS = [True, False]

N_EDGE_ROWS = 60


def _pick(values: list[Any], i: int, stride: int) -> Any:
    # every 7th row of each column is NULL, at a different offset per column
    if (i + stride) % 7 == 0:
        return None
    # 7 is coprime with every value list's length, so each value appears
    return values[(i * 7 + stride * 3) % len(values)]


def edge_rows(
    *,
    nan: bool = True,
    max_date: date = date(9999, 12, 31),
    max_timestamp: datetime = datetime(9999, 12, 31, 23, 59, 59, 999999),
) -> list[tuple[Any, ...]]:
    """
    Rows of the edge table.

    A database that can't store some values gets them as NULL (`nan=False`: MySQL can't
    store NaN or infinities) or clamped (ClickHouse dates end in 2299).
    """
    rows = []
    for i in range(N_EDGE_ROWS):
        f = _pick(F64, i, 5)
        if not nan and f is not None and (f != f or f in (float("inf"), float("-inf"))):
            f = None
        d = _pick(DATES, i, 9)
        if d is not None and d > max_date:
            d = max_date
        ts = _pick(TIMESTAMPS, i, 10)
        if ts is not None and ts > max_timestamp:
            ts = max_timestamp
        rows.append(
            (
                i,
                _pick(I16, i, 1),
                _pick(I32, i, 2),
                _pick(I64, i, 3),
                f,
                _pick(DEC, i, 4),
                _pick(STRINGS, i, 6),
                _pick(STRINGS, i, 8),
                d,
                ts,
                _pick(BOOLS, i, 11),
            )
        )
    return rows


EDGE_SCHEMA: dict[str, PolarsDataType] = {
    "id": pl.Int64,
    "i16": pl.Int16,
    "i32": pl.Int32,
    "i64": pl.Int64,
    "f64": pl.Float64,
    "dec": pl.Decimal(18, 4),
    "s": pl.String,
    "sb": pl.String,
    "d": pl.Date,
    "ts": pl.Datetime("us"),
    "b": pl.Boolean,
}


def edge_frame(**kwargs: Any) -> pl.DataFrame:
    return pl.DataFrame(edge_rows(**kwargs), schema=EDGE_SCHEMA, orient="row")


def _comparisons(col: str, literals: list[Any]) -> list[pl.Expr]:
    c = pl.col(col)
    out = []
    for lit in literals:
        lit_expr = pl.lit(lit)
        out += [
            c == lit_expr,
            c != lit_expr,
            c < lit_expr,
            c <= lit_expr,
            c > lit_expr,
            c >= lit_expr,
        ]
    return out


def _predicates() -> list[pl.Expr]:
    p: list[pl.Expr] = []
    # integers, at and around their type limits
    p += _comparisons("i16", [0, 32767, -32768])
    p += _comparisons("i32", [42, 2**31 - 1])
    p += _comparisons("i64", [0, 2**63 - 1, -(2**63)])
    p += [
        pl.col("i32").is_between(-42, 42),
        pl.col("i64").is_between(0, 10**12, closed="left"),
        pl.col("i16").is_in([0, 7, None]),
        pl.col("i16").is_in([0, 7, None], nulls_equal=True),
        pl.col("i64").is_in(list(range(5000))),
    ]
    # floats: NaN, signed zeros and infinities
    p += _comparisons("f64", [0.0, -0.0, 1.5])
    p += [
        pl.col("f64") > float("nan"),
        pl.col("f64") == float("nan"),
        pl.col("f64") < float("inf"),
        pl.col("f64").is_nan(),
        pl.col("f64").is_in([1.5, -1.5]),
    ]
    # decimals
    p += _comparisons("dec", [Decimal("12.5"), Decimal("0")])
    # strings, in the default and in a binary collation
    for col in ("s", "sb"):
        c = pl.col(col)
        p += [
            c == "abc",
            c != "abc",
            c == "ABC",
            c == "abc ",
            c == "",
            c != "",
            c < "b",
            c >= "B",
            c == "é",
            c == "ss",
            c.is_in(["abc", "é", "😀"]),
            c.str.starts_with("a"),
            c.str.starts_with("A"),
            c.str.ends_with(" "),
            c.str.contains("%", literal=True),
            c.str.contains("_", literal=True),
            c.str.contains("\\", literal=True),
            c.str.contains("'", literal=True),
            ~(c == "abc"),
            c.eq_missing("abc"),
            c.ne_missing("abc"),
        ]
    # dates and timestamps
    p += _comparisons("d", [date(1970, 1, 1), date(9999, 12, 31)])
    p += _comparisons(
        "ts", [datetime(1970, 1, 1, 0, 0, 0, 1), datetime(2026, 3, 29, 1, 30)]
    )
    p += [
        pl.col("ts") > pl.lit(datetime(1970, 1, 1)).cast(pl.Datetime("ns")),
        pl.col("d").is_between(date(1950, 1, 1), date(2030, 1, 1)),
    ]
    # booleans
    p += [
        pl.col("b"),
        ~pl.col("b"),
        pl.col("b") == False,  # noqa: E712
        pl.col("b").ne_missing(True),
    ]
    # NULL checks on every column
    for col in EDGE_COLUMNS[1:]:
        p += [pl.col(col).is_null(), pl.col(col).is_not_null()]
    # NOT, AND and OR, including the known traps
    p += [
        ~((pl.col("s") == "abc") | (pl.col("i16") > 0)),
        (pl.col("s") == "abc") & (pl.col("i32") > 0),
        (pl.col("s") == "abc") | pl.col("sb").is_null(),
        ~(pl.col("f64") > 5),
        ~(pl.col("s") < "b"),
        (pl.col("i16") > 0) & ~(pl.col("b")),
        ((pl.col("i16") > 0) | (pl.col("i32") < 0)) & (pl.col("d") > date(1970, 1, 1)),
        ~((pl.col("i64") == 0) & (pl.col("dec") > 0)),
        (pl.col("i16") * 2 > 10) & (pl.col("i32") > 0),
        ((pl.col("i16") + 1) == 1) | (pl.col("i32") == 42),
        ~(pl.col("s").str.starts_with("a") & pl.col("sb").is_not_null()),
        # found by the property tests: *_missing and nulls_equal are never NULL
        ~pl.col("f64").eq_missing(0.0),
        ~(pl.col("d").is_null() & pl.col("f64").eq_missing(0.0)),
        ~pl.col("i16").ne_missing(0),
        ~pl.col("s").is_in(["abc", "b"], nulls_equal=True),
        ~pl.col("s").is_in(["abc", None], nulls_equal=True),
    ]
    return p


PREDICATES: list[pl.Expr] = _predicates()


def ids(df: pl.DataFrame) -> list[int]:
    return sorted(df["id"].to_list())


def expected_ids(reference: pl.DataFrame, predicate: pl.Expr) -> list[int] | Exception:
    try:
        return ids(reference.filter(predicate))
    except Exception as e:
        return e


CORPUS_MODES = ["table", "query", "table-p4", "query-p4"]

FULL = bool(os.environ.get("POLARS_TEST_DATABASE_FULL"))

# Predicates a non-base cell runs although it plans them like the base cell: they check
# that the cell reads rows back, which the SQL comparison can't
SAMPLE = {"table": 4, "query": 4, "table-p4": 2, "query-p4": 2}


def corpus_for_mode(mode: str, *, full: bool = FULL) -> list[pl.Expr]:
    """The predicates a corpus mode runs: those at `corpus_indices`."""
    return [PREDICATES[i] for i in corpus_indices(mode, full=full)]


def corpus_indices(mode: str, *, full: bool = FULL) -> list[int]:
    """
    The positions in `PREDICATES` of the predicates a corpus mode runs.

    `"table"` and `"query"` run all of them: a query has no catalog (no declared types,
    collations or key), so its filters are classified on other information. The
    partitioned modes take disjoint shares: partitions only AND a range predicate to the
    same SQL. With `full`, the two shares cover the corpus (each predicate runs
    partitioned once); otherwise an eighth each. The split rotates every six
    predicates, so every operator (`_comparisons` emits six per literal) runs
    partitioned, in both modes in the full run.
    """
    if mode in ("table", "query"):
        return list(range(len(PREDICATES)))
    k = CORPUS_MODES.index(mode) - 2
    n = 2 if full else 8
    return [i for i in range(len(PREDICATES)) if (i + i // 6) % n == k]


def rendered_sql(lf: pl.LazyFrame, predicate: pl.Expr) -> str:
    """What `lf.filter(predicate)` pushes and the SQL it renders, without running it."""
    try:
        plan = lf.filter(predicate).select("id").explain()
    except Exception as e:
        return f"error: {type(e).__name__}"
    lines = (ln.strip() for ln in plan.splitlines())
    return "\n".join(ln for ln in lines if ln.startswith(("pushed:", "sql[")))


@dataclass(frozen=True)
class CorpusPlans:
    """
    What decides the rows each corpus predicate returns on one unpartitioned scan.

    The same server returns the same rows for the same SQL, and Polars filters what is
    left the same way given the same dtypes and values: two scans whose plans agree on
    a predicate return the same rows for it. A predicate pushed whole and exactly leaves
    nothing to Polars: then only the SQL and the ids decide.
    """

    # `rendered_sql` per predicate, with the scan's source as `<source>`: a table and a
    # query over it compare equal when only the source differs
    rendered: list[str]
    schema: pl.Schema
    # the whole table as the scan's connection reads it, sorted by id
    reference: pl.DataFrame

    @staticmethod
    def of(lf: pl.LazyFrame, reference: pl.DataFrame) -> CorpusPlans:
        bare = rendered_sql(lf, pl.lit(value=True))
        # the first FROM is the scan's own; a query source holds another
        source = bare.split(" FROM ", 1)[-1].split(" WHERE ", 1)[0]
        return CorpusPlans(
            [rendered_sql(lf, p).replace(source, "<source>") for p in PREDICATES],
            lf.collect_schema(),
            reference.sort("id"),
        )

    def same_columns(self, other: CorpusPlans) -> set[str]:
        """Columns both scans decode with the same dtype and values."""
        return {
            c
            for c in self.reference.columns
            if c in other.reference.columns
            and self.schema.get(c) == other.schema.get(c)
            and self.reference.schema[c] == other.reference.schema[c]
            and self.reference[c].equals(other.reference[c])
        }


def _pushed_exactly(rendered: str) -> bool:
    # "pushed: projection 2/11, filters 1/1 (1 exact), ..."
    filters = rendered.split("filters ", 1)[-1].split(",", 1)[0]
    done, _, rest = filters.partition("/")
    total, _, exact = rest.partition(" (")
    return done == total != "0" and exact.startswith(f"{done} exact")


def cell_predicates(
    mode: str,
    plans: CorpusPlans,
    base: CorpusPlans,
    *,
    sample: int,
    offset: int = 0,
) -> list[pl.Expr]:
    """
    The predicates of `corpus_for_mode(mode)` a cell runs, given a base that runs them.

    Those the cell plans like the base (see `CorpusPlans`) return the base's rows: of
    them, `sample` run, spread over the corpus (`offset` numbers the cell, so cells
    sample different ones). Every other predicate runs. Partitioned modes compare
    unpartitioned plans: partitions only add a range.
    """
    columns = plans.same_columns(base)

    def same(i: int) -> bool:
        if plans.rendered[i] != base.rendered[i] or "id" not in columns:
            return False
        return (
            _pushed_exactly(plans.rendered[i])
            or set(PREDICATES[i].meta.root_names()) <= columns
        )

    indices = corpus_indices(mode)
    flags = [same(i) for i in indices]
    n_same = sum(flags)
    picks = {
        (13 * offset + j * n_same // sample) % max(n_same, 1) for j in range(sample)
    }
    out, k = [], 0
    for i, s in zip(indices, flags, strict=True):
        if not s or k in picks:
            out.append(PREDICATES[i])
        k += s
    return out


def check_corpus(
    make_lf: Any,
    reference: pl.DataFrame,
    predicates: list[pl.Expr],
    *,
    check_exact: bool = False,
    max_failures: int = 20,
) -> tuple[list[str], dict[str, int]]:
    """
    Run each predicate on `make_lf()` and compare its ids with the Polars result.

    Returns the failures and pushdown counts. With `check_exact`, each predicate's plan
    is also inspected: a predicate pushed exactly isn't re-applied by Polars, so the
    comparison itself checks the exactness claim. Stops after `max_failures`: a broken
    cell fails in seconds, with enough examples to see why.
    """
    failures: list[str] = []
    stats = {"predicates": 0, "fully_exact": 0, "pushed": 0}
    for p in predicates:
        if len(failures) >= max_failures:
            failures.append(
                f"(stopped after {max_failures} failures; there may be more)"
            )
            break
        stats["predicates"] += 1
        expected = expected_ids(reference, p)
        lf = make_lf().filter(p)
        try:
            got: list[int] | Exception = ids(lf.select("id").collect())
        except Exception as e:
            got = e
        if isinstance(expected, Exception) or isinstance(got, Exception):
            if not (isinstance(expected, Exception) and isinstance(got, Exception)):
                failures.append(f"{p}: expected {expected!r}, got {got!r}")
            continue
        if got != expected:
            extra = sorted(set(got) - set(expected))
            missing = sorted(set(expected) - set(got))
            failures.append(f"{p}: extra ids {extra}, missing ids {missing}")
        if check_exact:
            plan = lf.explain()
            pushed = next((ln for ln in plan.splitlines() if "pushed:" in ln), "")
            if " 0/" not in pushed.split("filters", 1)[-1][:6]:
                stats["pushed"] += 1
            # "filters n/n (n exact)"
            part = pushed.split("filters ", 1)[-1].split(",", 1)[0]
            frac, _, exact = part.partition(" (")
            done, _, total = frac.partition("/")
            if done == total and exact.startswith(f"{done} exact") and done != "0":
                stats["fully_exact"] += 1
    return failures, stats


_LEAF_VALUES: dict[str, list[Any]] = {
    "i16": I16,
    "i32": I32,
    "i64": I64,
    "f64": [v for v in F64 if v == v],  # NaN literals are covered by the fixed corpus
    "dec": DEC,
    "s": STRINGS,
    "sb": STRINGS,
    "d": DATES,
    "ts": TIMESTAMPS,
}


def predicate_strategy(*, boolean_column: bool = True) -> Any:
    """
    Hypothesis: random predicate trees over the edge table, depth <= 4.

    `boolean_column=False` for databases without a boolean type (MySQL's BOOLEAN is an
    integer), where `b` can't be used as a predicate.
    """
    from hypothesis import strategies as st

    ops = [
        lambda c, v: c == v,
        lambda c, v: c != v,
        lambda c, v: c < v,
        lambda c, v: c <= v,
        lambda c, v: c > v,
        lambda c, v: c >= v,
        lambda c, v: c.eq_missing(v),
        lambda c, v: c.ne_missing(v),
    ]

    @st.composite
    def leaf(draw: Any) -> pl.Expr:
        col = draw(st.sampled_from(sorted(_LEAF_VALUES)))
        values = _LEAF_VALUES[col]
        kind = draw(st.integers(0, 9))
        c = pl.col(col)
        if kind == 0:
            return c.is_null()
        if kind == 1:
            return c.is_not_null()
        if kind == 2:
            items = draw(st.lists(st.sampled_from(values), min_size=1, max_size=4))
            return c.is_in(items)
        if kind == 3 and col in ("s", "sb"):
            s = draw(st.sampled_from(values))
            f = draw(st.sampled_from(["starts_with", "ends_with", "contains"]))
            if f == "contains":
                return c.str.contains(s, literal=True)
            expr: pl.Expr = getattr(c.str, f)(s)
            return expr
        if kind == 4 and col not in ("s", "sb"):
            lo, hi = sorted(
                draw(st.lists(st.sampled_from(values), min_size=2, max_size=2))
            )
            return c.is_between(lo, hi)
        if kind == 5 and boolean_column:
            return pl.col("b") if draw(st.booleans()) else ~pl.col("b")
        op = draw(st.sampled_from(ops))
        out: pl.Expr = op(c, pl.lit(draw(st.sampled_from(values))))
        return out

    return st.recursive(
        leaf(),
        lambda children: st.one_of(
            st.tuples(children, children).map(lambda t: t[0] & t[1]),
            st.tuples(children, children).map(lambda t: t[0] | t[1]),
            children.map(lambda c: ~c),
        ),
        max_leaves=6,
    )
