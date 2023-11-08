# Joins

## Join strategies

Polars supports the following join strategies by specifying the `strategy` argument:

| Strategy | Description                                                                                                                                                                                                |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `inner`  | Returns row with matching keys in _both_ frames. Non-matching rows in either the left or right frame are discarded.                                                                                        |
| `left`   | Returns all rows in the left dataframe, whether or not a match in the right-frame is found. Non-matching rows have their right columns null-filled.                                                        |
| `outer`  | Returns all rows from both the left and right dataframe. If no match is found in one frame, columns from the other frame are null-filled.                                                                  |
| `cross`  | Returns the Cartesian product of all rows from the left frame with all rows from the right frame. Duplicates rows are retained; the table length of `A` cross-joined with `B` is always `len(A) × len(B)`. |
| `asof`   | A left-join in which the match is performed on the _nearest_ key rather than on equal keys.                                                                                                                |
| `semi`   | Returns all rows from the left frame in which the join key is also present in the right frame.                                                                                                             |
| `anti`   | Returns all rows from the left frame in which the join key is _not_ present in the right frame.                                                                                                            |

### Inner join

An `inner` join produces a `DataFrame` that contains only the rows where the join key exists in both `DataFrames`. Let's take for example the following two `DataFrames`:

{{code_block('user-guide/transformations/joins','innerdf',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:setup"
--8<-- "python/user-guide/transformations/joins.py:innerdf"
```

<p></p>

{{code_block('user-guide/transformations/joins','innerdf2',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:innerdf2"
```

To get a `DataFrame` with the orders and their associated customer we can do an `inner` join on the `customer_id` column:

{{code_block('user-guide/transformations/joins','inner',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:inner"
```

### Left join

The `left` join produces a `DataFrame` that contains all the rows from the left `DataFrame` and only the rows from the right `DataFrame` where the join key exists in the left `DataFrame`. If we now take the example from above and want to have a `DataFrame` with all the customers and their associated orders (regardless of whether they have placed an order or not) we can do a `left` join:

{{code_block('user-guide/transformations/joins','left',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:left"
```

Notice, that the fields for the customer with the `customer_id` of `3` are null, as there are no orders for this customer.

### Outer join

The `outer` join produces a `DataFrame` that contains all the rows from both `DataFrames`. Columns are null, if the join key does not exist in the source `DataFrame`. Doing an `outer` join on the two `DataFrames` from above produces a similar `DataFrame` to the `left` join:

{{code_block('user-guide/transformations/joins','outer',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:outer"
```

### Cross join

A `cross` join is a cartesian product of the two `DataFrames`. This means that every row in the left `DataFrame` is joined with every row in the right `DataFrame`. The `cross` join is useful for creating a `DataFrame` with all possible combinations of the columns in two `DataFrames`. Let's take for example the following two `DataFrames`.

{{code_block('user-guide/transformations/joins','df3',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df3"
```

<p></p>

{{code_block('user-guide/transformations/joins','df4',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df4"
```

We can now create a `DataFrame` containing all possible combinations of the colors and sizes with a `cross` join:

{{code_block('user-guide/transformations/joins','cross',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:cross"
```

<br>

The `inner`, `left`, `outer` and `cross` join strategies are standard amongst dataframe libraries. We provide more details on the less familiar `semi`, `anti` and `asof` join strategies below.

### Semi join

The `semi` join returns all rows from the left frame in which the join key is also present in the right frame. Consider the following scenario: a car rental company has a `DataFrame` showing the cars that it owns with each car having a unique `id`.

{{code_block('user-guide/transformations/joins','df5',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df5"
```

The company has another `DataFrame` showing each repair job carried out on a vehicle.

{{code_block('user-guide/transformations/joins','df6',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df6"
```

You want to answer this question: which of the cars have had repairs carried out?

An inner join does not answer this question directly as it produces a `DataFrame` with multiple rows for each car that has had multiple repair jobs:

{{code_block('user-guide/transformations/joins','inner2',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:inner2"
```

However, a semi join produces a single row for each car that has had a repair job carried out.

{{code_block('user-guide/transformations/joins','semi',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:semi"
```

### Anti join

Continuing this example, an alternative question might be: which of the cars have **not** had a repair job carried out? An anti join produces a `DataFrame` showing all the cars from `df_cars` where the `id` is not present in the `df_repairs` `DataFrame`.

{{code_block('user-guide/transformations/joins','anti',['join'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:anti"
```

### Asof join

An `asof` join is like a left join except that we match on nearest key rather than equal keys.
In Polars we can do an asof join with the `join` method and specifying `strategy="asof"`. However, for more flexibility we can use the `join_asof` method.

Consider the following scenario: a stock market broker has a `DataFrame` called `df_trades` showing transactions it has made for different stocks.

{{code_block('user-guide/transformations/joins','df7',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df7"
```

The broker has another `DataFrame` called `df_quotes` showing prices it has quoted for these stocks.

{{code_block('user-guide/transformations/joins','df8',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:df8"
```

You want to produce a `DataFrame` showing for each trade the most recent quote provided _before_ the trade. You do this with `join_asof` (using the default `strategy = "backward"`).
To avoid joining between trades on one stock with a quote on another you must specify an exact preliminary join on the stock column with `by="stock"`.

{{code_block('user-guide/transformations/joins','asof',['join_asof'])}}

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:asofpre"
--8<-- "python/user-guide/transformations/joins.py:asof"
```

If you want to make sure that only quotes within a certain time range are joined to the trades you can specify the `tolerance` argument. In this case we want to make sure that the last preceding quote is within 1 minute of the trade so we set `tolerance = "1m"`.

=== ":fontawesome-brands-python: Python"

```python
--8<-- "python/user-guide/transformations/joins.py:asof2"
```

```python exec="on" result="text" session="user-guide/transformations/joins"
--8<-- "python/user-guide/transformations/joins.py:asof2"
```
