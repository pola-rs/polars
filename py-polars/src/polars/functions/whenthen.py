from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

import polars._reexport as pl
from polars._utils.parse import parse_predicates_constraints_into_expression

with contextlib.suppress(ImportError):  # Module not available when building docs
    import polars._plr as plr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from polars._typing import IntoExprColumn


def when(
    *predicates: IntoExprColumn | Iterable[IntoExprColumn] | bool,
    **constraints: Any,
) -> pl.When:
    """
    Start a `when-then-otherwise` expression.

    Always initiated by a `pl.when().then()`., and optionally followed by chaining one
    or more `.when().then()` statements.

    An optional `.otherwise()` can be appended at the end. If not declared, a default
    of `.otherwise(None)` is used.

    Similar to :func:`coalesce`, the value from the first condition that
    evaluates to True will be picked.

    If all conditions are False, the `otherwise` value is picked.

    Parameters
    ----------
    predicates
        Condition(s) that must be met in order to apply the subsequent statement.
        Accepts one or more boolean expressions, which are implicitly combined with
        `&`.
    constraints
        Apply conditions as `col_name = value` keyword arguments that are treated as
        equality matches, such as `x = 123`. As with the predicates parameter, multiple
        conditions are implicitly combined using `&`.

    Warnings
    --------
    Polars computes all expressions passed to `when-then-otherwise` in parallel and
    only applies the `when` conditions afterwards. This means each expression must be
    valid on its own, regardless of the conditions in the `when-then-otherwise` chain.

    Notes
    -----
    * String inputs e.g. `when("string")`, `then("string")` or `otherwise("string")`
      are parsed as column names. :func:`lit` can be used to create string values.
    * The expression output name is taken from the first `then` statement. It is
      not affected by `predicates`, nor by `constraints`.

    Examples
    --------
    Below we add a column with the value 1, where column "foo" > 2 and the value
    1 + column "bar" where it isn't.

    >>> df = pl.DataFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})
    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2).then(1).otherwise(1 + pl.col.bar).alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 4   │
    │ 3   ┆ 4   ┆ 1   │
    │ 4   ┆ 0   ┆ 1   │
    └─────┴─────┴─────┘

    A `when-then-otherwise` expression does not restrict the evaluation of the
    expressions passed to `then` and `otherwise` to the rows where the condition is
    true. In the example below, `pl.col("bar").sum()` is computed over the full
    column first, and the `when` condition is applied to the already-computed result.

    >>> agg_df = pl.DataFrame({"foo": ["a", "a", "b"], "bar": [2, 3, 4]})
    >>> agg_df.with_columns(
    ...     pl.when(pl.col("foo") == "a")
    ...     .then(pl.col("bar").sum())
    ...     .alias("sum")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬──────┐
    │ foo ┆ bar ┆ sum  │
    │ --- ┆ --- ┆ ---  │
    │ str ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ a   ┆ 2   ┆ 9    │
    │ a   ┆ 3   ┆ 9    │
    │ b   ┆ 4   ┆ null │
    └─────┴─────┴──────┘

    Note that `when-then` always executes all expressions.

    The results are folded left to right, picking the `then` value from the first `when`
    condition that is True.

    If no `when` condition is True the `otherwise` value is picked.

    >>> df.with_columns(
    ...     when = pl.col.foo > 2,
    ...     then = 1,
    ...     otherwise = 1 + pl.col.bar
    ... ).with_columns(
    ...     pl.when("when").then("then").otherwise("otherwise").alias("val")
    ... )
    shape: (3, 6)
    ┌─────┬─────┬───────┬──────┬───────────┬─────┐
    │ foo ┆ bar ┆ when  ┆ then ┆ otherwise ┆ val │
    │ --- ┆ --- ┆ ---   ┆ ---  ┆ ---       ┆ --- │
    │ i64 ┆ i64 ┆ bool  ┆ i32  ┆ i64       ┆ i64 │
    ╞═════╪═════╪═══════╪══════╪═══════════╪═════╡
    │ 1   ┆ 3   ┆ false ┆ 1    ┆ 4         ┆ 4   │
    │ 3   ┆ 4   ┆ true  ┆ 1    ┆ 5         ┆ 1   │
    │ 4   ┆ 0   ┆ true  ┆ 1    ┆ 1         ┆ 1   │
    └─────┴─────┴───────┴──────┴───────────┴─────┘

    Note that in regular Polars usage, a single string is parsed as a column name.

    >>> df.with_columns(
    ...     when = pl.col.foo > 2,
    ...     then = "foo",
    ...     otherwise = "bar"
    ... )
    shape: (3, 5)
    ┌─────┬─────┬───────┬──────┬───────────┐
    │ foo ┆ bar ┆ when  ┆ then ┆ otherwise │
    │ --- ┆ --- ┆ ---   ┆ ---  ┆ ---       │
    │ i64 ┆ i64 ┆ bool  ┆ i64  ┆ i64       │
    ╞═════╪═════╪═══════╪══════╪═══════════╡
    │ 1   ┆ 3   ┆ false ┆ 1    ┆ 3         │
    │ 3   ┆ 4   ┆ true  ┆ 3    ┆ 4         │
    │ 4   ┆ 0   ┆ true  ┆ 4    ┆ 0         │
    └─────┴─────┴───────┴──────┴───────────┘

    For consistency, `when-then` behaves in the same way.

    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2).then("foo").otherwise("bar").alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 3   │
    │ 3   ┆ 4   ┆ 3   │
    │ 4   ┆ 0   ┆ 4   │
    └─────┴─────┴─────┘

    :func:`lit` can be used to create string values.

    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2)
    ...     .then(pl.lit("foo"))
    ...     .otherwise(pl.lit("bar"))
    ...     .alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ str │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ bar │
    │ 3   ┆ 4   ┆ foo │
    │ 4   ┆ 0   ┆ foo │
    └─────┴─────┴─────┘

    Multiple `when-then` statements can be chained.

    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2)
    ...     .then(1)
    ...     .when(pl.col.bar > 2)
    ...     .then(4)
    ...     .otherwise(-1)
    ...     .alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i32 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 4   │
    │ 3   ┆ 4   ┆ 1   │
    │ 4   ┆ 0   ┆ 1   │
    └─────┴─────┴─────┘

    In the case of `foo=3` and `bar=4`, both conditions are True but the first value
    (i.e. 1) is picked.

    >>> df.with_columns(
    ...     when1 = pl.col.foo > 2,
    ...     then1 = 1,
    ...     when2 = pl.col.bar > 2,
    ...     then2 = 4,
    ...     otherwise = -1
    ... )
    shape: (3, 7)
    ┌─────┬─────┬───────┬───────┬───────┬───────┬───────────┐
    │ foo ┆ bar ┆ when1 ┆ then1 ┆ when2 ┆ then2 ┆ otherwise │
    │ --- ┆ --- ┆ ---   ┆ ---   ┆ ---   ┆ ---   ┆ ---       │
    │ i64 ┆ i64 ┆ bool  ┆ i32   ┆ bool  ┆ i32   ┆ i32       │
    ╞═════╪═════╪═══════╪═══════╪═══════╪═══════╪═══════════╡
    │ 1   ┆ 3   ┆ false ┆ 1     ┆ true  ┆ 4     ┆ -1        │
    │ 3   ┆ 4   ┆ true  ┆ 1     ┆ true  ┆ 4     ┆ -1        │
    │ 4   ┆ 0   ┆ true  ┆ 1     ┆ false ┆ 4     ┆ -1        │
    └─────┴─────┴───────┴───────┴───────┴───────┴───────────┘

    The `otherwise` statement is optional and defaults to `.otherwise(None)`
    if not given.

    This idiom is commonly used to null out values.

    >>> df.with_columns(pl.when(pl.col.foo == 3).then("bar"))
    shape: (3, 2)
    ┌─────┬──────┐
    │ foo ┆ bar  │
    │ --- ┆ ---  │
    │ i64 ┆ i64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    │ 3   ┆ 4    │
    │ 4   ┆ null │
    └─────┴──────┘

    `when` accepts keyword arguments as shorthand for equality conditions.

    >>> df.with_columns(pl.when(foo=3).then("bar"))
    shape: (3, 2)
    ┌─────┬──────┐
    │ foo ┆ bar  │
    │ --- ┆ ---  │
    │ i64 ┆ i64  │
    ╞═════╪══════╡
    │ 1   ┆ null │
    │ 3   ┆ 4    │
    │ 4   ┆ null │
    └─────┴──────┘

    Multiple predicates passed to `when` are combined with `&`

    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2, pl.col.bar < 3) # when((pred1) & (pred2))
    ...     .then(pl.lit("Yes"))
    ...     .otherwise(pl.lit("No"))
    ...     .alias("val")
    ... )
    shape: (3, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ val │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ str │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ No  │
    │ 3   ┆ 4   ┆ No  │
    │ 4   ┆ 0   ┆ Yes │
    └─────┴─────┴─────┘

    It could also be thought of as an implicit :func:`all_horizontal` being present.

    >>> df.with_columns(
    ...     when = pl.all_horizontal(pl.col.foo > 2, pl.col.bar < 3)
    ... )
    shape: (3, 3)
    ┌─────┬─────┬───────┐
    │ foo ┆ bar ┆ when  │
    │ --- ┆ --- ┆ ---   │
    │ i64 ┆ i64 ┆ bool  │
    ╞═════╪═════╪═══════╡
    │ 1   ┆ 3   ┆ false │
    │ 3   ┆ 4   ┆ false │
    │ 4   ┆ 0   ┆ true  │
    └─────┴─────┴───────┘

    Structs can be used as a way to return multiple values.

    Here we swap the "foo" and "bar" values when "foo" is greater than 2.

    >>> df.with_columns(
    ...     pl.when(pl.col.foo > 2)
    ...     .then(pl.struct(foo="bar", bar="foo"))
    ...     .otherwise(pl.struct("foo", "bar"))
    ...     .struct.unnest()
    ... )
    shape: (3, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 1   ┆ 3   │
    │ 4   ┆ 3   │
    │ 0   ┆ 4   │
    └─────┴─────┘

    The struct fields are given the same name as the target columns, which are then
    unnested.

    >>> df.with_columns(
    ...     when = pl.col.foo > 2,
    ...     then = pl.struct(foo="bar", bar="foo"),
    ...     otherwise = pl.struct("foo", "bar")
    ... )
    shape: (3, 5)
    ┌─────┬─────┬───────┬───────────┬───────────┐
    │ foo ┆ bar ┆ when  ┆ then      ┆ otherwise │
    │ --- ┆ --- ┆ ---   ┆ ---       ┆ ---       │
    │ i64 ┆ i64 ┆ bool  ┆ struct[2] ┆ struct[2] │
    ╞═════╪═════╪═══════╪═══════════╪═══════════╡
    │ 1   ┆ 3   ┆ false ┆ {3,1}     ┆ {1,3}     │
    │ 3   ┆ 4   ┆ true  ┆ {4,3}     ┆ {3,4}     │
    │ 4   ┆ 0   ┆ true  ┆ {0,4}     ┆ {4,0}     │
    └─────┴─────┴───────┴───────────┴───────────┘

    The output name of a `when-then` expression comes from the first `then` branch.

    Here we try to set all columns to 0 if any column contains a value less than 2.

    >>> df.with_columns( # doctest: +SKIP
    ...    pl.when(pl.any_horizontal(pl.all() < 2))
    ...    .then(0)
    ...    .otherwise(pl.all())
    ... )
    # ComputeError: the name 'literal' passed to `LazyFrame.with_columns` is duplicate

    :meth:`.name.keep` could be used to give preference to the column expression.

    >>> df.with_columns(
    ...    pl.when(pl.any_horizontal(pl.all() < 2))
    ...    .then(0)
    ...    .otherwise(pl.all())
    ...    .name.keep()
    ... )
    shape: (3, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 0   ┆ 0   │
    │ 3   ┆ 4   │
    │ 0   ┆ 0   │
    └─────┴─────┘

    The logic could also be changed to move the column expression inside `then`.

    >>> df.with_columns(
    ...     pl.when(pl.any_horizontal(pl.all() < 2).not_())
    ...     .then(pl.all())
    ...     .otherwise(0)
    ... )
    shape: (3, 2)
    ┌─────┬─────┐
    │ foo ┆ bar │
    │ --- ┆ --- │
    │ i64 ┆ i64 │
    ╞═════╪═════╡
    │ 0   ┆ 0   │
    │ 3   ┆ 4   │
    │ 0   ┆ 0   │
    └─────┴─────┘
    """  # fmt: skip
    condition = parse_predicates_constraints_into_expression(*predicates, **constraints)
    return pl.When(plr.when(condition))
