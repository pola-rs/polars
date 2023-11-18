# Aggregation

Polars implements a powerful syntax defined not only in its lazy API, but also in its eager API. Let's take a look at what that means.

We can start with the simple [US congress `dataset`](https://github.com/unitedstates/congress-legislators).

{{code_block('user-guide/expressions/aggregation','dataframe',['DataFrame','Categorical'])}}

#### Basic aggregations

You can easily combine different aggregations by adding multiple expressions in a
`list`. There is no upper bound on the number of aggregations you can do, and you can
make any combination you want. In the snippet below we do the following aggregations:

Per GROUP `"first_name"` we

<!-- dprint-ignore-start -->

- count the number of rows in the group:
    - short form: `pl.count("party")`
    - full form: `pl.col("party").count()`
- aggregate the gender values groups:
    - full form: `pl.col("gender")`
- get the first value of column `"last_name"` in the group:
    - short form: `pl.first("last_name")` (not available in Rust)
    - full form: `pl.col("last_name").first()`

<!-- dprint-ignore-end -->

Besides the aggregation, we immediately sort the result and limit to the top `5` so that
we have a nice summary overview.

{{code_block('user-guide/expressions/aggregation','basic',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:setup"
--8<-- "python/user-guide/expressions/aggregation.py:dataframe"
--8<-- "python/user-guide/expressions/aggregation.py:basic"
```

#### Conditionals

It's that easy! Let's turn it up a notch. Let's say we want to know how
many delegates of a "state" are "Pro" or "Anti" administration. We could directly query
that in the aggregation without the need of a `lambda` or grooming the `DataFrame`.

{{code_block('user-guide/expressions/aggregation','conditional',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:conditional"
```

Similarly, this could also be done with a nested GROUP BY, but that doesn't help show off some of these nice features. 😉

{{code_block('user-guide/expressions/aggregation','nested',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:nested"
```

#### Filtering

We can also filter the groups. Let's say we want to compute a mean per group, but we
don't want to include all values from that group, and we also don't want to filter the
rows from the `DataFrame` (because we need those rows for another aggregation).

In the example below we show how this can be done.

!!! note

    Note that we can make Python functions for clarity. These functions don't cost us anything. That is because we only create Polars expressions, we don't apply a custom function over a `Series` during runtime of the query. Of course, you can make functions that return expressions in Rust, too.

{{code_block('user-guide/expressions/aggregation','filter',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:filter"
```

#### Sorting

It's common to see a `DataFrame` being sorted for the sole purpose of managing the ordering during a GROUP BY operation. Let's say that we want to get the names of the oldest and youngest politicians per state. We could SORT and GROUP BY.

{{code_block('user-guide/expressions/aggregation','sort',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort"
```

However, **if** we also want to sort the names alphabetically, this breaks. Luckily we can sort in a `group_by` context separate from the `DataFrame`.

{{code_block('user-guide/expressions/aggregation','sort2',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort2"
```

We can even sort by another column in the `group_by` context. If we want to know if the alphabetically sorted name is male or female we could add: `pl.col("gender").sort_by("first_name").first().alias("gender")`

{{code_block('user-guide/expressions/aggregation','sort3',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort3"
```

### Do not kill parallelization

!!! warning "Python Users Only"

    The following section is specific to Python, and doesn't apply to Rust. Within Rust, blocks and closures (lambdas) can, and will, be executed concurrently.

We have all heard that Python is slow, and does "not scale." Besides the overhead of
running "slow" bytecode, Python has to remain within the constraints of the Global
Interpreter Lock (GIL). This means that if you were to use a `lambda` or a custom Python
function to apply during a parallelized phase, Polars speed is capped running Python
code preventing any multiple threads from executing the function.

This all feels terribly limiting, especially because we often need those `lambda` functions in a
`.group_by()` step, for example. This approach is still supported by Polars, but
keeping in mind bytecode **and** the GIL costs have to be paid. It is recommended to try to solve your queries using the expression syntax before moving to `lambdas`. If you want to learn more about using `lambdas`, go to the [user defined functions section](./user-defined-functions.md).

### Conclusion

In the examples above we've seen that we can do a lot by combining expressions. By doing so we delay the use of custom Python functions that slow down the queries (by the slow nature of Python AND the GIL).

If we are missing a type expression let us know by opening a
[feature request](https://github.com/pola-rs/polars/issues/new/choose)!
