# Joins

A join operation combines columns from one or more dataframes into a new dataframe. The different
“joining strategies” and matching criteria used by the different types of joins influence how
columns are combined and also what rows are included in the result of the join operation.

The most common type of join is an “equi join”, in which rows are matched by a key expression.
Polars supports several joining strategies for equi joins, which determine exactly how we handle the
matching of rows. Polars also supports “non-equi joins”, a type of join where the matching criterion
is not an equality, and a type of join where rows are matched by key proximity, called “asof join”.

## Quick reference table

The table below acts as a quick reference for people who know what they are looking for. If you want
to learn about joins in general and how to work with them in Polars, feel free to skip the table and
keep reading below.

=== ":fontawesome-brands-python: Python"

    [:material-api: `join`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join.html)
    [:material-api: `join_where`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_where.html)
    [:material-api: `join_asof`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.join_asof.html)

=== ":fontawesome-brands-rust: Rust"

    [:material-api: `join`](https://docs.pola.rs/api/rust/dev/polars/prelude/trait.DataFrameJoinOps.html#method.join)
    ([:material-flag-plus: semi_anti_join](/user-guide/installation/#feature-flags "Enable the feature flag semi_anti_join for semi and for anti joins"){.feature-flag} needed for some options.)
    [:material-api: `join_asof_by`](https://docs.pola.rs/api/rust/dev/polars/prelude/trait.AsofJoinBy.html#method.join_asof_by)
    [:material-flag-plus: Available on feature asof_join](/user-guide/installation/#feature-flags "To use this functionality enable the feature flag asof_join"){.feature-flag}
    [:material-api: `join_where`](https://docs.rs/polars/latest/polars/prelude/struct.JoinBuilder.html#method.join_where)
    [:material-flag-plus: Available on feature iejoin](/user-guide/installation/#feature-flags "To use this functionality enable the feature flag iejoin"){.feature-flag}

| Type                  | Function                   | Brief description                                                                                                                                                       |
| --------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Equi inner join       | `join(..., how="inner")`   | Keeps rows that matched both on the left and right.                                                                                                                     |
| Equi left outer join  | `join(..., how="left")`    | Keeps all rows from the left plus matching rows from the right. Non-matching rows from the left have their right columns filled with `null`.                            |
| Equi right outer join | `join(..., how="right")`   | Keeps all rows from the right plus matching rows from the left. Non-matching rows from the right have their left columns filled with `null`.                            |
| Equi full join        | `join(..., how="full")`    | Keeps all rows from either dataframe, regardless of whether they match or not. Non-matching rows from one side have the columns from the other side filled with `null`. |
| Equi semi join        | `join(..., how="semi")`    | Keeps rows from the left that have a match on the right.                                                                                                                |
| Equi anti join        | `join(..., how="anti")`    | Keeps rows from the left that do not have a match on the right.                                                                                                         |
| Non-equi inner join   | `join_where`               | Finds all possible pairings of rows from the left and right that satisfy the given predicate(s).                                                                        |
| Asof join             | `join_asof`/`join_asof_by` | Like a left outer join, but matches on the nearest key instead of on exact key matches.                                                                                 |
| Cartesian product     | `join(..., how="cross")`   | Computes the [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of the two dataframes.                                                                |

## Equi joins

In an equi join, rows are matched by checking equality of a key expression. You can do an equi join
with the function `join` by specifying the name of the column to be used as key. For the examples,
we will be loading some (modified) Monopoly property data.

First, we load a dataframe that contains property names and their colour group in the game:

{{code_block('user-guide/transformations/joins','props_groups',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:prep-data"
--8<-- "python/user-guide/transformations/joins.py:props_groups"
```

Next, we load a dataframe that contains property names and their price in the game:

{{code_block('user-guide/transformations/joins','props_prices',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:props_prices"
```

Now, we join both dataframes to create a dataframe that contains property names, colour groups, and
prices:

{{code_block('user-guide/transformations/joins','equi-join',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:equi-join"
```

The result has four rows but both dataframes used in the operation had five rows. Polars uses a
joining strategy to determine what happens with rows that have multiple matches or with rows that
have no match at all. By default, Polars computes an “inner join” but there are
[other join strategies that we show next](#join-strategies).

In the example above, the two dataframes conveniently had the column we wish to use as key with the
same name and with the values in the exact same format. Suppose, for the sake of argument, that one
of the dataframes had a differently named column and the other had the property names in lower case:

{{code_block('user-guide/transformations/joins','props_groups2',['Expr.str'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:props_groups2"
```

{{code_block('user-guide/transformations/joins','props_prices2',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:props_prices2"
```

In a situation like this, where we may want to perform the same join as before, we can leverage
`join`'s flexibility and specify arbitrary expressions to compute the joining key on the left and on
the right, allowing one to compute row keys dynamically:

{{code_block('user-guide/transformations/joins', 'join-key-expression', ['join', 'Expr.str'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:join-key-expression"
```

Because we are joining on the right with an expression, Polars preserves the column “property_name”
from the left and the column “name” from the right so we can have access to the original values that
the key expressions were applied to.

## Join strategies

When computing a join with `df1.join(df2, ...)`, we can specify one of many different join
strategies. A join strategy specifies what rows to keep from each dataframe based on whether they
match rows from the other dataframe.

### Inner join

In an inner join the resulting dataframe only contains the rows from the left and right dataframes
that matched. That is the default strategy used by `join` and above we can see an example of that.
We repeat the example here and explicitly specify the join strategy:

{{code_block('user-guide/transformations/joins','inner-join',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:inner-join"
```

The result does not include the row from `props_groups` that contains “The Shire” and the result
also does not include the row from `props_prices` that contains “Sesame Street”.

### Left join

A left outer join is a join where the result contains all the rows from the left dataframe and the
rows of the right dataframe that matched any rows from the left dataframe.

{{code_block('user-guide/transformations/joins','left-join',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:left-join"
```

If there are any rows from the left dataframe that have no matching rows on the right dataframe,
they get the value `null` on the new columns.

### Right join

Computationally speaking, a right outer join is exactly the same as a left outer join, but with the
arguments swapped. Here is an example:

{{code_block('user-guide/transformations/joins','right-join',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:right-join"
```

We show that `df1.join(df2, how="right", ...)` is the same as `df2.join(df1, how="left", ...)`, up
to the order of the columns of the result, with the computation below:

{{code_block('user-guide/transformations/joins','left-right-join-equals',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:left-right-join-equals"
```

### Full join

A full outer join will keep all of the rows from the left and right dataframes, even if they don't
have matching rows in the other dataframe:

{{code_block('user-guide/transformations/joins','full-join',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:full-join"
```

In this case, we see that we get two columns `property_name` and `property_name_right` to make up
for the fact that we are matching on the column `property_name` of both dataframes and there are
some names for which there are no matches. The two columns help differentiate the source of each row
data. If we wanted to force `join` to coalesce the two columns `property_name` into a single column,
we could set `coalesce=True` explicitly:

{{code_block('user-guide/transformations/joins','full-join-coalesce',['join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:full-join-coalesce"
```

When not set, the parameter `coalesce` is determined automatically from the join strategy and the
key(s) specified, which is why the inner, left, and right, joins acted as if `coalesce=True`, even
though we didn't set it.

### Semi join

A semi join will return the rows of the left dataframe that have a match in the right dataframe, but
we do not actually join the matching rows:

{{code_block('user-guide/transformations/joins', 'semi-join', [], ['join'],
['join-semi_anti_join_flag'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:semi-join"
```

A semi join acts as a sort of row filter based on a second dataframe.

### Anti join

Conversely, an anti join will return the rows of the left dataframe that do not have a match in the
right dataframe:

{{code_block('user-guide/transformations/joins', 'anti-join', [], ['join'],
['join-semi_anti_join_flag'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:anti-join"
```

## Non-equi joins

In a non-equi join matches between the left and right dataframes are computed differently. Instead
of looking for matches on key expressions, we provide a single predicate that determines what rows
of the left dataframe can be paired up with what rows of the right dataframe.

For example, consider the following Monopoly players and their current cash:

{{code_block('user-guide/transformations/joins','players',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:players"
```

Using a non-equi join we can easily build a dataframe with all the possible properties that each
player could be interested in buying. We use the function `join_where` to compute a non-equi join:

{{code_block('user-guide/transformations/joins','non-equi',['join_where'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:non-equi"
```

You can provide multiple expressions as predicates, in that case they will be AND combined. You can
also combine expressions in a single expression if you need other combinations like OR or XOR.

## Asof join

An `asof` join is like a left join except that we match on nearest key rather than equal keys. In
Polars we can do an asof join with the `join_asof` method.

For the asof join we will consider a scenario inspired by the stock market. Suppose a stock market
broker has a dataframe called `df_trades` showing transactions it has made for different stocks.

{{code_block('user-guide/transformations/joins','df_trades',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df_trades"
```

The broker has another dataframe called `df_quotes` showing prices it has quoted for these stocks:

{{code_block('user-guide/transformations/joins','df_quotes',[])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df_quotes"
```

You want to produce a dataframe showing for each trade the most recent quote provided _on or before_
the time of the trade. You do this with `join_asof` (using the default `strategy = "backward"`). To
avoid joining between trades on one stock with a quote on another you must specify an exact
preliminary join on the stock column with `by="stock"`.

{{code_block('user-guide/transformations/joins','asof', [], ['join_asof'], ['join_asof_by'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:asof"
```

If you want to make sure that only quotes within a certain time range are joined to the trades you
can specify the `tolerance` argument. In this case we want to make sure that the last preceding
quote is within 1 minute of the trade so we set `tolerance = "1m"`.

{{code_block('user-guide/transformations/joins','asof-tolerance', [], ['join_asof'],
['join_asof_by'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:asof-tolerance"
```

## Cartesian product

Polars allows you to compute the
[Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) of two dataframes, producing a
dataframe where all rows of the left dataframe are paired up with all the rows of the right
dataframe. To compute the Cartesian product of two dataframes, you can pass the strategy
`how="cross"` to the function `join` without specifying any of `on`, `left_on`, and `right_on`:

{{code_block('user-guide/transformations/joins','cartesian-product',[],['join'],['cross_join'])}}

```python exec="on" result="text" session="transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:cartesian-product"
```
