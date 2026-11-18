from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def ctx() -> pl.SQLContext[pl.DataFrame]:
    t = pl.DataFrame(
        {"g": [1, 1, 2, 2, 3], "x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, None]}
    )
    return pl.SQLContext(t=t, eager=True)


def rows(ctx: pl.SQLContext[pl.DataFrame], query: str) -> list[tuple[object, ...]]:
    return ctx.execute(query).rows()


def test_scalar_macros(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    assert rows(ctx, "CREATE MACRO add1(a) AS a + 1") == [("CREATE MACRO add1",)]
    ctx.execute("CREATE TEMP MACRO mysum(a) AS sum(a)")
    ctx.execute("CREATE FUNCTION twice(a) AS a * 2")
    ctx.execute("CREATE MACRO coalz(a, b := 0) AS coalesce(a, b)")
    ctx.execute("CREATE MACRO ov(a) AS a, (a, b) AS a + b")

    assert rows(ctx, "SELECT ADD1(x) AS r FROM t ORDER BY r LIMIT 2") == [(2,), (3,)]
    assert rows(ctx, "SELECT twice(1 + 2), ov(1), ov(1, 2)") == [(6, 1, 3)]
    assert rows(
        ctx,
        "SELECT g, add1(mysum(x)) AS s FROM t GROUP BY g HAVING mysum(x) > 3 ORDER BY g",
    ) == [(2, 8), (3, 6)]
    assert rows(ctx, "SELECT mysum(x) FROM t") == [(15,)]
    assert rows(
        ctx,
        "SELECT x FROM t QUALIFY add1(ROW_NUMBER() OVER (ORDER BY x DESC)) <= 3 ORDER BY x",
    ) == [(4,), (5,)]
    assert rows(
        ctx,
        "SELECT coalz(y) AS a, coalz(y, b := -1) AS b FROM t WHERE x > 3 ORDER BY x",
    ) == [(40, 40), (0, -1)]
    assert rows(ctx, "SELECT coalz(b := 2, a := NULL)") == [(2,)]
    assert rows(
        ctx, "SELECT x FROM t WHERE add1(x) IN (SELECT twice(x) FROM t) ORDER BY x"
    ) == [(1,), (3,), (5,)]

    # a macro can call other macros, as they are at the time of the call
    ctx.execute("CREATE MACRO m(a) AS twice(add1(a))")
    assert rows(ctx, "SELECT m(1)") == [(4,)]
    ctx.execute("CREATE OR REPLACE MACRO add1(a) AS a + 10")
    assert rows(ctx, "SELECT m(1)") == [(22,)]

    ctx.execute("DROP MACRO add1")
    ctx.execute("DROP FUNCTION IF EXISTS add1")
    with pytest.raises(SQLInterfaceError, match="unsupported function 'add1'"):
        ctx.execute("SELECT add1(1)")


def test_macro_subquery(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute("CREATE MACRO max_y(a) AS (SELECT max(y) FROM t WHERE x <= a)")
    assert rows(ctx, "SELECT max_y(3)") == [(30,)]
    with pytest.raises(SQLInterfaceError, match="its value cannot read columns"):
        ctx.execute("SELECT max_y(x) FROM t")


def test_table_macros(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute(
        "CREATE MACRO rng(lo, hi := 10) AS TABLE SELECT * FROM t WHERE x BETWEEN lo AND hi"
    )
    ctx.execute("CREATE MACRO first_n(n) AS TABLE SELECT x FROM rng(1, n)")
    ctx.execute("CREATE MACRO val(v) AS TABLE SELECT v")

    assert rows(ctx, "SELECT x FROM rng(2, 3) ORDER BY x") == [(2,), (3,)]
    assert rows(ctx, "SELECT count(*) FROM rng(4)") == [(2,)]
    assert rows(ctx, "SELECT rng.x FROM rng(lo := 5)") == [(5,)]
    assert rows(
        ctx, "SELECT r.x, t.y FROM rng(2, 3) AS r JOIN t ON r.x = t.x ORDER BY r.x"
    ) == [(2, 20), (3, 30)]
    assert rows(ctx, "SELECT count(*) FROM first_n(3)") == [(3,)]
    assert rows(ctx, "SELECT * FROM val(7)") == [(7,)]
    assert ctx.execute("SELECT * FROM val(7)").columns == ["v"]

    # scalar and table macros have their own names
    ctx.execute("CREATE MACRO val(v) AS v + 1")
    assert rows(ctx, "SELECT val(1), * FROM val(1)") == [(2, 1)]
    ctx.execute("DROP MACRO TABLE val")
    assert rows(ctx, "SELECT val(1)") == [(2,)]

    with pytest.raises(SQLInterfaceError, match="its value cannot read columns"):
        ctx.execute("SELECT * FROM t, rng(t.x)")


def test_macro_depth(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    for i in range(80):
        body = f"m{i + 1}(a)" if i < 79 else "a"
        ctx.execute(f"CREATE MACRO m{i}(a) AS {body}")
    assert rows(ctx, "SELECT m40(1)") == [(1,)]
    with pytest.raises(SQLInterfaceError, match="more than 64 macros deep"):
        ctx.execute("SELECT m0(1)")

    for i in range(5):
        body = f"n{i + 1}(a)" if i < 4 else "a"
        ctx.execute(f"CREATE MACRO n{i}(a) AS {'abs(' * 15}{body}{')' * 15}")
    assert rows(ctx, "SELECT n3(-1)") == [(1,)]
    with pytest.raises(SQLInterfaceError, match="nested too deeply"):
        ctx.execute("SELECT n0(-1)")

    for i in range(30):
        body = f"SELECT * FROM q{i + 1}()" if i < 29 else "SELECT 1 AS a"
        ctx.execute(f"CREATE MACRO q{i}() AS TABLE {body}")
    assert rows(ctx, "SELECT * FROM q20()") == [(1,)]
    with pytest.raises(SQLInterfaceError, match="nested too deeply"):
        ctx.execute("SELECT * FROM q0()")

    # the same limit as for the SQL written out
    ctx.execute("CREATE MACRO inc(a) AS 1 + a")
    assert rows(ctx, "SELECT " + "inc(" * 10 + "0" + ")" * 10) == [(10,)]
    with pytest.raises(SQLInterfaceError, match="nested too deeply"):
        ctx.execute("SELECT " + "inc(" * 24 + "0" + ")" * 24)
    ctx.execute("CREATE MACRO identity(a) AS a")
    assert rows(ctx, "SELECT 1" + "::INT" * 48 + ", identity(0)") == [(1, 0)]


def test_macro_arguments(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute("CREATE MACRO ov(b) AS b + 100, (a, b) AS a + b")
    assert rows(ctx, "SELECT ov(1, b := 2)") == [(3,)]
    assert rows(ctx, "SELECT ov(b := 2)") == [(102,)]

    ctx.execute("CREATE MACRO fld(a) AS a.v")
    assert rows(ctx, "SELECT fld(s) FROM (SELECT {'v': 7} AS s) t") == [(7,)]
    # the field of the argument, not a column of a table with the same name
    res = rows(ctx, "SELECT fld(p) FROM (SELECT {'v': 7} AS p, 99 AS v) AS p")
    assert res == [(7,)]
    # field names are not parameters
    ctx.execute("CREATE MACRO value_or(obj, value) AS COALESCE((obj).value, value)")
    res = rows(ctx, "SELECT value_or(p, f) FROM (SELECT {'value': 7} AS p, 0 AS f)")
    assert res == [(7,)]
    query = """
        SELECT g, fld({'v': SUM(x)}) AS a, SUM(y) AS v
        FROM (VALUES (1, 10, 100), (1, 20, 200), (2, 40, 400)) t(g, x, y)
        GROUP BY g ORDER BY g
    """
    assert rows(ctx, query) == [(1, 30, 300), (2, 40, 400)]
    query = """
        SELECT g, fld({'v': g + 1}) AS a, SUM(y) AS v
        FROM (VALUES (1, 10), (1, 20), (2, 40)) t(g, y)
        GROUP BY g ORDER BY g
    """
    assert rows(ctx, query) == [(1, 2, 30), (2, 3, 40)]
    # a qualifier naming a relation of the body is not the parameter
    ctx.execute("""
        CREATE MACRO cheaper(p) AS TABLE
        SELECT p.price FROM (VALUES (10), (30)) p(price) WHERE p.price < (p).price
    """)
    assert rows(ctx, "SELECT * FROM cheaper({'price': 20})") == [(10,)]
    ctx.execute("""
        CREATE OR REPLACE MACRO cheaper(p) AS TABLE
        SELECT p.price
        FROM ((VALUES (1, 10), (2, 30)) p(id, price) JOIN (VALUES (1), (2)) i(id) USING (id))
        WHERE p.price < (p).price
    """)
    assert rows(ctx, "SELECT * FROM cheaper({'price': 20})") == [(10,)]
    query = """
        SELECT {'v': -s.v} AS s FROM (VALUES ({'v': 1}), ({'v': 2})) t(s)
        GROUP BY s ORDER BY fld(s) LIMIT 1
    """
    assert rows(ctx, query) == [({"v": -1},)]
    ctx.execute("""
        CREATE MACRO prices(p) AS TABLE
        SELECT j.price, j.tax
        FROM ((VALUES (1, 10), (2, 30)) p(id, price)
              JOIN (VALUES (1, 100), (2, 200)) i(id, tax) ON p.id = i.id) j
        WHERE j.price < (p).max_price
    """)
    query = "SELECT * FROM prices({'id': 1, 'max_price': 50}) ORDER BY price"
    assert rows(ctx, query) == [(10, 100), (30, 200)]
    ctx.register("ns.items", pl.DataFrame({"price": [10, 30]}))
    ctx.execute("""
        CREATE MACRO below(ns) AS TABLE
        SELECT ns.items.price FROM ns.items WHERE ns.items.price < (ns).max_price
    """)
    query = "SELECT * FROM below({'items': {'price': 5}, 'max_price': 20})"
    assert rows(ctx, query) == [(10,)]
    ctx.execute("""
        CREATE MACRO cheapest(p) AS TABLE
        (SELECT p.price FROM (VALUES (30), (10), (20)) p(price)
         WHERE p.price < (p).max_price)
        ORDER BY p.price LIMIT 1
    """)
    assert rows(ctx, "SELECT * FROM cheapest({'price': 0, 'max_price': 40})") == [(10,)]

    ctx.execute("CREATE MACRO ids(p := [4]) AS TABLE SELECT * FROM UNNEST(p) AS u(id)")
    assert rows(ctx, "SELECT * FROM ids([1, 2])") == [(1,), (2,)]
    assert rows(ctx, "SELECT * FROM ids()") == [(4,)]
    ctx.execute("CREATE MACRO pair(a, b) AS TABLE SELECT * FROM UNNEST([a, b]) u(id)")
    assert rows(ctx, "SELECT * FROM pair(-1, 2)") == [(-1,), (2,)]
    ctx.execute("CREATE MACRO two() AS [1, 2]")
    assert rows(ctx, "SELECT * FROM UNNEST(two()) AS u(id)") == [(1,), (2,)]
    ctx.execute("CREATE MACRO date_fmt() AS '%Y-%m-%d'")
    res = rows(ctx, "SELECT STRPTIME('2026-01-02', date_fmt())")
    assert res == [(datetime(2026, 1, 2),)]
    # an inner relation named like the parameter only hides it in its own query
    ctx.execute("CREATE MACRO getv(p) AS p.v + (SELECT p.v FROM (VALUES (10)) p(v))")
    query = "SELECT getv(q) FROM (VALUES ({'v': 100}, {'v': 2})) t(p, q)"
    assert rows(ctx, query) == [(12,)]
    # a struct field is not a column
    ctx.execute("CREATE MACRO shift(d, span) AS d + span")
    res = rows(ctx, "SELECT shift(DATE '2026-01-31', INTERVAL '1' MONTH)")
    assert res == [(date(2026, 2, 28),)]
    ctx.execute("CREATE MACRO const(v) AS TABLE SELECT v AS value")
    query = """
        SELECT t.id, m.value FROM (VALUES (1), (2)) t(id)
        CROSS JOIN LATERAL const(3) AS m ORDER BY id
    """
    assert rows(ctx, query) == [(1, 3), (2, 3)]
    ctx.execute("CREATE MACRO wrap(p) AS TABLE SELECT p.value FROM LATERAL const(10) p")
    assert rows(ctx, "SELECT * FROM wrap({'value': 999})") == [(10,)]
    # a number from a macro is not a position
    ctx.execute("CREATE MACRO one() AS 1")
    query = (
        "SELECT x, y FROM (VALUES (1, 20), (2, 10)) t(x, y) ORDER BY one(), y LIMIT 1"
    )
    assert rows(ctx, query) == [(2, 10)]
    ctx.execute("""
        CREATE MACRO pick_by(k) AS (
            SELECT x FROM (VALUES (1, 20), (2, 10)) t(x, y) ORDER BY k, y LIMIT 1
        )
    """)
    assert rows(ctx, "SELECT pick_by(one())") == [(2,)]
    assert rows(ctx, "SELECT pick_by(1)") == [(1,)]
    ctx.execute(
        "CREATE MACRO pick_t(k) AS TABLE SELECT x FROM (VALUES (1, 20), (2, 10)) t(x, y) "
        "ORDER BY k, y LIMIT 1"
    )
    ctx.execute("CREATE MACRO wrap_t(k) AS TABLE SELECT * FROM pick_t(k)")
    assert rows(ctx, "SELECT * FROM wrap_t(one())") == [(2,)]
    ctx.execute("CREATE MACRO day_span(n) AS INTERVAL (n) DAY")
    res = rows(ctx, "SELECT DATE '2026-01-01' + day_span((one()))")
    assert res == [(date(2026, 1, 2),)]
    # each SELECT of a set operation has its own relations
    ctx.execute("""
        CREATE MACRO both_v(p) AS TABLE
        SELECT p.v AS v UNION ALL SELECT p.v FROM (VALUES (10)) p(v)
    """)
    assert rows(ctx, "SELECT * FROM both_v({'v': 2}) ORDER BY v") == [(2,), (10,)]
    # a CTE does not see the relations of the query around it
    ctx.execute("""
        CREATE MACRO below_cte(p) AS TABLE
        WITH f AS (SELECT price FROM (VALUES (10), (30)) s(price) WHERE price < p.max)
        SELECT p.price FROM f p
    """)
    assert rows(ctx, "SELECT * FROM below_cte({'max': 20})") == [(10,)]
    ctx.execute("CREATE MACRO no_key() AS NULL")
    query = "SELECT x FROM (VALUES (2), (1)) t(x) ORDER BY no_key(), x"
    assert rows(ctx, query) == [(1,), (2,)]
    ctx.execute("CREATE MACRO clean(s) AS LOWER(NORMALIZE(TRIM(s), NFKC))")
    assert rows(ctx, "SELECT clean('  \uff21lice ')") == [("alice",)]
    ctx.execute("CREATE MACRO first_n(n) AS TABLE SELECT x FROM t ORDER BY x LIMIT n")
    assert rows(ctx, "SELECT * FROM first_n(2)") == [(1,), (2,)]
    ctx.execute("CREATE MACRO int_list(x) AS [CAST(x AS INT)]")
    assert rows(ctx, "SELECT int_list(-1)") == [([-1],)]
    ctx.execute("CREATE MACRO int_list2(x) AS [x::INT]")
    assert rows(ctx, "SELECT int_list2(-1)") == [([-1],)]
    ctx.execute("CREATE MACRO day_key() AS '2026-01-01'")
    query = "SELECT d FROM (VALUES (DATE '2026-01-01')) t(d) WHERE d IN (day_key())"
    assert rows(ctx, query) == [(date(2026, 1, 1),)]
    ctx.execute("CREATE MACRO width() AS 2")
    ctx.execute("CREATE MACRO bounds(w) AS [-w, w]")
    assert rows(ctx, "SELECT bounds(width())") == [([-2, 2],)]
    query = "SELECT x FROM (VALUES (2), (1)) t(x) ORDER BY -width(), x"
    assert rows(ctx, query) == [(1,), (2,)]
    # a CTE of a subquery reads the relations around that subquery (not supported yet)
    ctx.execute("""
        CREATE MACRO selected(p) AS TABLE
        SELECT p.id FROM (VALUES (1), (2)) p(id)
        WHERE EXISTS (WITH c AS (SELECT p.id AS id) SELECT * FROM c WHERE id = (p).id)
    """)
    with pytest.raises(SQLInterfaceError, match="EXISTS subquery"):
        rows(ctx, "SELECT * FROM selected({'id': 1})")
    # a qualified struct argument stays one expression
    ctx.execute("CREATE MACRO customer_id(c) AS c.id")
    ctx.execute("CREATE TABLE orders AS SELECT {'id': 7} AS customer")
    ctx.execute("CREATE TABLE crm.customer AS SELECT 99 AS id")
    query = """
        SELECT customer_id(crm.customer) AS id
        FROM orders AS crm CROSS JOIN crm.customer
    """
    assert rows(ctx, query) == [(7,)]
    query = """
        SELECT o.order_id, (
            SELECT c.name FROM (VALUES (7, 'Alice'), (8, 'Bob')) c(id, name)
            WHERE c.id = customer_id(o.customer)
        ) AS name
        FROM (VALUES (1, {'id': 7}), (2, {'id': 8})) o(order_id, customer)
        ORDER BY order_id
    """
    assert rows(ctx, query) == [(1, "Alice"), (2, "Bob")]
    ctx.execute("CREATE MACRO width_cast() AS CAST(2 AS INT)")
    assert rows(ctx, "SELECT [-width_cast(), width_cast()]") == [([-2, 2],)]
    ctx.execute("CREATE MACRO limit_value(options) AS options.n")
    query = "SELECT x FROM (VALUES (3), (1), (2)) t(x) ORDER BY x LIMIT limit_value({'n': 2})"
    assert rows(ctx, query) == [(1,), (2,)]
    query = """
        SELECT t.id, (
            SELECT u.v FROM (VALUES (1, 10), (1, 20), (2, 30)) u(id, v)
            WHERE u.id = t.id ORDER BY u.v LIMIT one()
        ) AS v
        FROM (VALUES (1), (2)) t(id) ORDER BY t.id
    """
    assert rows(ctx, query) == [(1, 10), (2, 30)]
    query = """
        SELECT t.id, (
            SELECT u.x FROM (VALUES (1, 10, 20), (1, 20, 10)) u(id, x, y)
            WHERE u.id = t.id ORDER BY one(), u.y LIMIT 1
        ) AS picked
        FROM (VALUES (1)) t(id)
    """
    assert rows(ctx, query) == [(1, 20)]
    query = """
        SELECT a.id, (
            SELECT COUNT(*) FROM (VALUES (1), (2)) b(id)
            WHERE b.id < (
                SELECT c.v FROM (VALUES (1, 10), (1, 20), (2, 30)) c(id, v)
                WHERE c.id = a.id ORDER BY one(), c.v DESC LIMIT 1
            )
        ) AS n
        FROM (VALUES (1), (2)) a(id) ORDER BY a.id
    """
    assert rows(ctx, query) == [(1, 2), (2, 2)]
    ctx.execute("CREATE MACRO quarter_key() AS 'q1'")
    query = """
        SELECT q1 FROM (VALUES ('q1', 10), ('q2', 20)) sales(quarter, amount)
        PIVOT (SUM(amount) FOR quarter IN (quarter_key()))
    """
    assert rows(ctx, query) == [(10,)]
    # an aggregate of outer values only is not one of the subquery
    ctx.execute("CREATE MACRO field_value(s) AS s.v")
    query = """
        SELECT EXISTS (SELECT MAX(field_value(t.s)) FROM (VALUES (0)) u(v) WHERE FALSE)
        FROM (VALUES ({'v': 1})) t(s) GROUP BY t.s
    """
    with pytest.raises(SQLInterfaceError):
        rows(ctx, query)
    # a CTE in a CTE does not see the relations of either query
    ctx.execute("""
        CREATE MACRO nested_scope(p) AS TABLE
        WITH c AS (
            WITH d AS (SELECT p.max AS v FROM (VALUES ({'max': 99})) data(p))
            SELECT * FROM d
        )
        SELECT p.v FROM c p
    """)
    assert rows(ctx, "SELECT * FROM nested_scope({'max': 20})") == [(20,)]
    ctx.execute("CREATE MACRO str_width() AS CAST('2' AS INT)")
    query = "SELECT x FROM (VALUES (-2), (2)) t(x) WHERE x IN (-str_width())"
    assert rows(ctx, query) == [(-2,)]
    ctx.execute(
        "CREATE MACRO value_or(obj, value) AS TABLE "
        "SELECT COALESCE((obj).value, value) AS v"
    )
    assert rows(ctx, "SELECT * FROM value_or({'value': 7}, 0)") == [(7,)]
    query = """
        SELECT t.id, (
            SELECT fld(u.s) FROM (VALUES (1, {'v': 7}), (1, {'v': 7}), (2, {'v': 8})) u(id, s)
            WHERE u.id = t.id GROUP BY u.s.v
        ) AS v
        FROM (VALUES (1), (2)) t(id) ORDER BY t.id
    """
    assert rows(ctx, query) == [(1, 7), (2, 8)]
    ctx.execute("CREATE MACRO month_span() AS INTERVAL '1' MONTH")
    res = rows(ctx, "SELECT DATE '2026-01-31' + month_span()")
    assert res == [(date(2026, 2, 28),)]
    query = """
        SELECT t.id, (
            SELECT fld(u.s) FROM (VALUES (1, {'v': 7}), (2, {'v': 8})) u(id, s)
            WHERE u.id = t.id
        ) FROM (VALUES (1), (2)) t(id) ORDER BY t.id
    """
    assert rows(ctx, query) == [(1, 7), (2, 8)]
    query = """
        SELECT t.id, (
            SELECT fld(s) FROM (VALUES (1, {'v': 7}, 99), (2, {'v': 8}, 98)) s(id, s, v)
            WHERE s.id = t.id GROUP BY s.s, s.v
        ) AS v
        FROM (VALUES (1), (2)) t(id) ORDER BY t.id
    """
    assert rows(ctx, query) == [(1, 7), (2, 8)]
    query = """
        SELECT t.id, (
            SELECT fld(u.s.n) FROM (VALUES (1, {'n': {'v': 7}})) u(id, s)
            WHERE u.id = t.id GROUP BY u.s.n.v
        ) AS v
        FROM (VALUES (1)) t(id)
    """
    assert rows(ctx, query) == [(1, 7)]
    assert rows(ctx, query.replace("fld(u.s.n)", "fld((u.s).n)")) == [(1, 7)]
    query = query.replace("GROUP BY u.s.n.v", "GROUP BY fld((u.s).n)")
    assert rows(ctx, query) == [(1, 7)]
    query = """
        SELECT t.id, (
            SELECT fld(s) FROM (VALUES (1, {'v': 7}), (1, {'v': 7}), (2, {'v': 8})) u(id, s)
            WHERE u.id = t.id GROUP BY u.s.v
        ) AS v
        FROM (VALUES (1), (2)) t(id) ORDER BY t.id
    """
    assert rows(ctx, query) == [(1, 7), (2, 8)]
    ctx.execute("CREATE MACRO lt(n) AS TABLE SELECT x FROM t WHERE x < n")
    assert rows(ctx, "SELECT * FROM lt(({'max': 3}).max) ORDER BY x") == [(1,), (2,)]

    # a view over a macro with a quantified subquery
    ctx.execute("""
        CREATE MACRO matches(n) AS TABLE
        SELECT t.x, t.x = ANY (
            SELECT u.x FROM (VALUES (1), (3)) u(x) WHERE u.x = t.x AND u.x < n
        ) AS present
        FROM (VALUES (1), (2), (3)) t(x)
    """)
    ctx.execute("CREATE VIEW v AS SELECT * FROM matches(3)")
    expected = [(1, True), (2, False), (3, False)]
    assert rows(ctx, "SELECT * FROM v ORDER BY x") == expected

    query = """
        SELECT {'v': -s.v} AS s FROM (VALUES ({'v': 1}), ({'v': 2})) t(s)
        GROUP BY s ORDER BY fld(t.s) LIMIT 1
    """
    assert rows(ctx, query) == [({"v": -1},)]
    ctx.execute('CREATE MACRO star(a) AS a."*"')
    assert rows(ctx, "SELECT star({'a': 1, '*': 3})") == [(3,)]

    ctx.execute("CREATE MACRO pick(p) AS (SELECT p FROM t LIMIT 1)")
    with pytest.raises(SQLInterfaceError, match="where its value cannot read columns"):
        ctx.execute("SELECT pick(COLUMNS('^x$')) FROM t")
    ctx.execute("CREATE MACRO colx() AS COLUMNS('^x$')")
    ctx.execute("CREATE MACRO tpick(p) AS TABLE SELECT p AS v FROM t")
    with pytest.raises(SQLInterfaceError, match="where its value cannot read columns"):
        ctx.execute("SELECT * FROM tpick(colx())")

    ctx.execute(
        "CREATE MACRO vals(p) AS TABLE (SELECT p) UNION ALL BY NAME (SELECT 10 AS p)"
    )
    assert rows(ctx, "SELECT SUM(p) FROM vals(7)") == [(17,)]

    with pytest.raises(
        SQLInterfaceError, match=r"default of parameter 'b'.* cannot read"
    ):
        ctx.execute("CREATE MACRO f(a, b := a + 1) AS a + b")
    ctx.execute("CREATE MACRO g(a, b := colx()) AS a + b")
    with pytest.raises(
        SQLInterfaceError, match=r"default of parameter 'b'.* cannot read"
    ):
        ctx.execute("SELECT g(1) FROM t")

    ctx.execute("CREATE MACRO choice(a, b := 10) AS a + b, (a) AS a + 100")
    with pytest.raises(SQLInterfaceError, match="more than one of its definitions"):
        ctx.execute("SELECT choice(1)")
    assert rows(ctx, "SELECT choice(1, 2)") == [(3,)]


def test_macro_order_by_position(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute(
        "CREATE MACRO first_by(k) AS TABLE SELECT x, y FROM t ORDER BY k DESC NULLS LAST LIMIT 1"
    )
    assert rows(ctx, "SELECT * FROM first_by(1)") == [(5, None)]
    assert rows(ctx, "SELECT * FROM first_by(2)") == [(4, 40)]


def test_table_macro_modifiers(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute("CREATE MACRO vals() AS TABLE SELECT 1 AS x")
    with pytest.raises(SQLInterfaceError, match="SAMPLE"):
        ctx.execute("SELECT * FROM vals() TABLESAMPLE BERNOULLI (0 PERCENT)")


def test_macro_view_random(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute("CREATE VIEW r AS SELECT RANDOM() AS x")
    ctx.execute("CREATE MACRO via() AS TABLE SELECT * FROM r")
    ctx.execute("CREATE VIEW v AS SELECT * FROM via()")
    res = ctx.execute("SELECT a.x = b.x AS same FROM v a CROSS JOIN v b")
    assert res.rows() == [(False,)]


def test_macro_in_view(ctx: pl.SQLContext[pl.DataFrame]) -> None:
    ctx.execute("CREATE MACRO add1(a) AS a + 1")
    ctx.execute("CREATE VIEW v AS SELECT add1(x) AS r FROM t WHERE x < 3")
    assert rows(ctx, "SELECT r FROM v ORDER BY r") == [(2,), (3,)]
    # the view reads the macro as it is now
    ctx.execute("CREATE OR REPLACE MACRO add1(a) AS a + 100")
    assert rows(ctx, "SELECT r FROM v ORDER BY r") == [(101,), (102,)]


@pytest.mark.parametrize(
    ("query", "match"),
    [
        ("CREATE MACRO add1(a) AS a + 2", "macro 'add1' already exists"),
        ("CREATE MACRO abs(a) AS a", "cannot create macro 'abs'"),
        ("CREATE MACRO trim(a) AS upper(a)", "cannot create macro 'trim'"),
        ("CREATE MACRO unnest(a) AS TABLE SELECT a", "cannot create macro 'unnest'"),
        (
            "CREATE MACRO read_csv(a) AS TABLE SELECT a",
            "cannot create macro 'read_csv'",
        ),
        ("CREATE MACRO f(a) AS a + x", r"can only use its parameters; found 'x'"),
        ("CREATE MACRO f(a, a) AS a", "duplicate parameter 'a'"),
        ("CREATE MACRO f(a := 1, b) AS a", "without a default follows"),
        ("CREATE MACRO s.f(a) AS a", "cannot be qualified"),
        ("SELECT add1()", "no definition for the given arguments"),
        ("SELECT add1(1, 2)", "no definition for the given arguments"),
        ("SELECT add1(b := 1)", "no definition for the given arguments"),
        ("SELECT add1(1, a := 2)", "parameter 'a' is given twice"),
        ("SELECT add1(x) OVER () FROM t", "does not support OVER"),
        ("SELECT add1(DISTINCT x) FROM t", "without DISTINCT"),
        ("SELECT rec(1)", "macro 'rec' calls itself"),
        ("SELECT dbl(1)", "macro 'dbl' calls itself"),
        ("DROP MACRO nope", "macro 'nope' does not exist"),
        ("DROP MACRO TABLE add1", "table macro 'add1' does not exist"),
    ],
)
def test_macro_errors(ctx: pl.SQLContext[pl.DataFrame], query: str, match: str) -> None:
    ctx.execute("CREATE MACRO add1(a) AS a + 1")
    ctx.execute("CREATE MACRO rec(a) AS rec(a)")
    ctx.execute("CREATE MACRO dbl(a) AS dbl(a + a)")
    with pytest.raises(SQLInterfaceError, match=match):
        ctx.execute(query)


def test_macro_file_reads(io_files_path: Path) -> None:
    ctx = pl.SQLContext()
    ctx.execute("CREATE MACRO csvs(paths) AS TABLE SELECT * FROM read_csv(paths)")
    path = (io_files_path / "foods1.csv").as_posix()
    assert ctx.execute(f"SELECT COUNT(*) FROM csvs(['{path}'])").collect().item() == 27
    ctx.execute(f"CREATE MACRO input_path() AS '{path}'")
    query = "SELECT COUNT(*) FROM read_csv([input_path()])"
    assert ctx.execute(query).collect().item() == 27
    ctx.execute("CREATE MACRO missing() AS 'NA'")
    query = f"SELECT COUNT(*) FROM read_csv('{path}', null_values := [missing()])"
    assert ctx.execute(query).collect().item() == 27


def test_macro_positional_join() -> None:
    ctx = pl.SQLContext()
    for body in ("a POSITIONAL JOIN b", "a AS l POSITIONAL JOIN b AS r"):
        ctx.execute(f"""
            CREATE OR REPLACE MACRO zipped() AS TABLE
            WITH a(x) AS (VALUES (1), (2)), b(y) AS (VALUES (10), (20), (30))
            SELECT * FROM {body}
        """)
        res = ctx.execute("SELECT * FROM zipped()", eager=True).rows()
        assert res == [(1, 10), (2, 20), (None, 30)]


def test_macro_expansion_size() -> None:
    ctx = pl.SQLContext()
    ctx.execute("CREATE MACRO m0(x) AS x + x")
    for k in range(1, 40):
        ctx.execute(f"CREATE MACRO m{k}(x) AS m{k - 1}(x) + m{k - 1}(x)")
    with pytest.raises(SQLInterfaceError, match="too large after expanding"):
        ctx.execute("SELECT m39(1) AS a", eager=True)


def test_macro_qualifier_of_relation() -> None:
    # `ns.v` is the field of the argument: `ns` alone does not name `ns.items`.
    ctx = pl.SQLContext({"ns.items": pl.DataFrame({"ns": [{"v": 7}]})})
    ctx.execute("CREATE MACRO m(ns) AS TABLE SELECT ns.v FROM ns.items")
    res = ctx.execute("SELECT * FROM m({'v': 99})", eager=True)
    assert res.to_dicts() == [{"v": 99}]


def test_macro_call_in_argument_of_same_macro() -> None:
    ctx = pl.SQLContext()
    ctx.execute("CREATE MACRO echo(v) AS TABLE SELECT v")
    res = ctx.execute("SELECT * FROM echo((SELECT 1 FROM echo(2)))", eager=True)
    assert res.rows() == [(1,)]


def test_macro_expansion_size_of_table_macros() -> None:
    ctx = pl.SQLContext()
    ctx.execute("CREATE MACRO t0() AS TABLE SELECT 1 AS x")
    for k in range(1, 40):
        ctx.execute(
            f"CREATE MACRO t{k}() AS TABLE SELECT * FROM t{k - 1}() a, t{k - 1}() b"
        )
    with pytest.raises(SQLInterfaceError, match="too large after expanding"):
        ctx.execute("SELECT * FROM t39()", eager=True)


def test_macro_rejects_query_params() -> None:
    ctx = pl.SQLContext()
    for sql in [
        "CREATE MACRO fixed() AS $1",
        "CREATE MACRO fixed(x := ?) AS x",
        "CREATE MACRO fixed() AS TABLE SELECT $v AS a",
    ]:
        with pytest.raises(SQLInterfaceError, match="not supported in CREATE MACRO"):
            ctx.execute(sql, params=[5])
    ctx.execute("CREATE MACRO f() AS 1")
    with pytest.raises(SQLInterfaceError, match=r"\$unused is not used"):
        ctx.execute("DROP MACRO f", params={"unused": 5})
    with pytest.raises(SQLInterfaceError, match=r"\$1 is not used"):
        ctx.execute("CREATE MACRO g() AS 1", params=[5])
    assert ctx.execute("SELECT f() AS a", eager=True).rows() == [(1,)]


def test_macro_lambda_inside_lambda() -> None:
    ctx = pl.SQLContext()
    ctx.execute("CREATE MACRO inc_all(xs) AS list_transform(xs, lambda e: e + 1)")
    res = ctx.execute(
        "SELECT list_transform([[1, 2], [3]], lambda row: inc_all(row)) AS r",
        eager=True,
    )
    assert res.to_dicts() == [{"r": [[2, 3], [4]]}]

    # A lambda in an argument or default that lands in a lambda of the body.
    ctx.execute("CREATE MACRO wrap(xs) AS list_transform([0], lambda z: xs)")
    res = ctx.execute(
        "SELECT wrap(list_transform([1, 2], lambda y: y + 1)) AS r", eager=True
    )
    assert res.to_dicts() == [{"r": [[2, 3]]}]
    ctx.execute(
        "CREATE MACRO wrap_default(xs := list_transform([1, 2], lambda y: y + 1)) "
        "AS list_transform([0], lambda z: xs)"
    )
    res = ctx.execute(
        "SELECT list_transform([5], lambda q: wrap_default()) AS r", eager=True
    )
    assert res.to_dicts() == [{"r": [[[2, 3]]]}]


def test_macro_alias_of_parenthesized_join() -> None:
    ctx = pl.SQLContext()
    ctx.execute(
        "CREATE MACRO pick(p) AS TABLE SELECT p.v "
        "FROM ((VALUES ({'v': 7})) p(p) CROSS JOIN (VALUES (1)) b(k)) j"
    )
    res = ctx.execute("SELECT * FROM pick({'v': 99})", eager=True)
    assert res.to_dicts() == [{"v": 99}]

    # The alias does not name a relation inside the join.
    ctx.execute(
        "CREATE MACRO pick_on(j) AS TABLE SELECT j.x "
        "FROM ((VALUES (1), (2)) a(x) JOIN (VALUES (0)) b(y) ON a.x = j.v) j"
    )
    res = ctx.execute("SELECT * FROM pick_on({'v': 2})", eager=True)
    assert res.rows() == [(2,)]


def test_table_macro_overloads() -> None:
    ctx = pl.SQLContext()
    ctx.execute(
        "CREATE MACRO t(v) AS TABLE SELECT v AS x, (v, w) AS TABLE SELECT v + w AS x"
    )
    assert ctx.execute("SELECT * FROM t(2)", eager=True).rows() == [(2,)]
    assert ctx.execute("SELECT * FROM t(2, 3)", eager=True).rows() == [(5,)]
    # Without a query after `AS TABLE`, it is an item of the SELECT.
    ctx.execute(
        "CREATE MACRO u(v) AS TABLE SELECT v AS x, (v) AS tbl FROM (VALUES (1))"
    )
    assert ctx.execute("SELECT * FROM u(4)", eager=True).rows() == [(4, 4)]


def test_table_macro_overload_boundaries() -> None:
    ctx = pl.SQLContext({"t": pl.DataFrame({"a": [7]})})
    ctx.execute("CREATE MACRO f(v) AS TABLE SELECT v, () AS TABLE FROM t")
    assert ctx.execute("SELECT * FROM f(3)", eager=True).rows() == [(3,)]
    assert ctx.execute("SELECT * FROM f()", eager=True).rows() == [(7,)]
    # A derived table aliased `table` is part of the body.
    ctx.execute(
        "CREATE MACRO g() AS TABLE "
        "SELECT * FROM (VALUES (1)) a(a), (VALUES (2)) AS table(b)"
    )
    assert ctx.execute("SELECT * FROM g()", eager=True).rows() == [(1, 2)]
    # The longest query that parses is the body.
    ctx.execute("CREATE MACRO one() AS TABLE SELECT 1 AS x, (a) AS table FROM t")
    assert ctx.execute("SELECT * FROM one()", eager=True).rows() == [(1, 7)]
    ctx.execute(
        "CREATE MACRO three(a) AS TABLE SELECT a, "
        "(a, b) AS TABLE SELECT a + b, () AS TABLE SELECT 7"
    )
    assert ctx.execute("SELECT * FROM three(1)", eager=True).rows() == [(1,)]
    assert ctx.execute("SELECT * FROM three(1, 2)", eager=True).rows() == [(3,)]
    assert ctx.execute("SELECT * FROM three()", eager=True).rows() == [(7,)]
    ctx.execute("CREATE MACRO h() AS TABLE SELECT /*+ some_hint */ 1 AS one")
    with pytest.raises(SQLInterfaceError, match="optimizer hints are not supported"):
        ctx.execute("SELECT * FROM h()", eager=True)


def test_table_macro_implicit_lateral() -> None:
    ctx = pl.SQLContext()
    # `p` names the relation before the derived table, not the parameter.
    ctx.execute(
        "CREATE MACRO m(p) AS TABLE "
        "SELECT d.x FROM (VALUES (1), (2)) p(v), (SELECT p.v AS x) d"
    )
    assert ctx.execute("SELECT * FROM m({'v': 99}) ORDER BY x", eager=True).rows() == [
        (1,),
        (2,),
    ]
    ctx.execute(
        "CREATE MACRO m2(p) AS TABLE "
        "SELECT d.x FROM (SELECT p.v AS x) d, (VALUES (1), (2)) p(v)"
    )
    assert ctx.execute("SELECT * FROM m2({'v': 99})", eager=True).rows() == [
        (99,),
        (99,),
    ]
    ctx.execute(
        "CREATE MACRO m3(p) AS TABLE "
        "SELECT d.x FROM LATERAL (SELECT p.v AS x) d, (VALUES (1), (2)) p(v)"
    )
    assert ctx.execute("SELECT * FROM m3({'v': 99})", eager=True).rows() == [
        (99,),
        (99,),
    ]
    ctx.execute(
        "CREATE MACRO m4(p) AS TABLE SELECT j.x "
        "FROM ((VALUES (1), (2)) p(v) CROSS JOIN (SELECT p.v AS x) d) j"
    )
    res = ctx.execute("SELECT * FROM m4({'v': 99}) ORDER BY x", eager=True)
    assert res.rows() == [(1,), (2,)]


def test_table_macro_join_scopes() -> None:
    ctx = pl.SQLContext()
    # A semi join drops its right side, so `p` reads the argument after the join.
    ctx.execute(
        "CREATE MACRO m(p) AS TABLE SELECT p.v "
        "FROM (VALUES (1, {'v': 7})) a(k, p) SEMI JOIN (VALUES (1)) p(v) ON a.k = p.v"
    )
    assert ctx.execute("SELECT * FROM m({'v': 99})", eager=True).rows() == [(99,)]
    # A relation joined later is not seen by an earlier join condition.
    ctx.execute(
        "CREATE MACRO m2(p) AS TABLE SELECT a.x "
        "FROM (VALUES (1), (2)) a(x) JOIN (VALUES (0)) b(y) ON a.x = p.v "
        "CROSS JOIN (VALUES (0)) p(v)"
    )
    assert ctx.execute("SELECT * FROM m2({'v': 2})", eager=True).rows() == [(2,)]
    # The dropped side of a semi join is not seen by later joins or derived tables.
    semi = (
        "FROM (VALUES (1, {'v': 7})) a(k, p) SEMI JOIN (VALUES (1)) p(v) ON a.k = p.v"
    )
    ctx.execute(
        f"CREATE MACRO m3(p) AS TABLE SELECT b.x {semi} "
        "JOIN (VALUES (7), (99)) b(x) ON b.x = p.v"
    )
    assert ctx.execute("SELECT * FROM m3({'v': 99})", eager=True).rows() == [(99,)]
    ctx.execute(
        f"CREATE MACRO m4(p) AS TABLE SELECT d.x {semi} CROSS JOIN (SELECT p.v AS x) d"
    )
    assert ctx.execute("SELECT * FROM m4({'v': 99})", eager=True).rows() == [(99,)]
    # A join condition in a LATERAL subquery sees the relations before it.
    ctx.execute(
        "CREATE MACRO m5(p) AS TABLE SELECT d.x FROM (VALUES (1), (2)) p(v), "
        "LATERAL (SELECT b.x FROM (VALUES (0)) a(k) "
        "JOIN (VALUES (1), (2)) b(x) ON b.x = p.v) d"
    )
    res = ctx.execute("SELECT * FROM m5({'v': 99}) ORDER BY x", eager=True)
    assert res.rows() == [(1,), (2,)]
    # The alias of a table function does not name a relation in its arguments.
    ctx.execute("CREATE MACRO seq(p) AS TABLE SELECT * FROM range(p.n) AS p(x)")
    res = ctx.execute("SELECT * FROM seq({'n': 3}) ORDER BY x", eager=True)
    assert res.rows() == [(0,), (1,), (2,)]
