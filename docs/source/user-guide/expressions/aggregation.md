# Aggregation

The Polars [context](../concepts/expressions-and-contexts.md#contexts) `group_by` lets you apply
expressions on subsets of columns, as defined by the unique values of the column over which the data
is grouped. This is a very powerful capability that we explore in this section of the user guide.

We start by reading in a
[US congress `dataset`](https://github.com/unitedstates/congress-legislators):

{{code_block('user-guide/expressions/aggregation','dataframe',['DataFrame','Categorical'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:dataframe"
```

## Basic aggregations

You can easily apply multiple expressions to your aggregated values. Simply list all of the
expressions you want inside the function `agg`. There is no upper bound on the number of
aggregations you can do and you can make any combination you want. In the snippet below we will
group the data based on the column “first_name” and then we will apply the following aggregations:

- count the number of rows in the group (which means we count how many people in the data set have
  each unique first name);
- combine the values of the column “gender” into a list by referring the column but omitting an
  aggregate function; and
- get the first value of the column “last_name” within the group.

After computing the aggregations, we immediately sort the result and limit it to the top five rows
so that we have a nice summary overview:

{{code_block('user-guide/expressions/aggregation','basic',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:basic"
```

It's that easy! Let's turn it up a notch.

## Conditionals

Let's say we want to know how many delegates of a state are “Pro” or “Anti” administration. We can
query that directly in the aggregation without the need for a `lambda` or grooming the dataframe:

{{code_block('user-guide/expressions/aggregation','conditional',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:conditional"
```

## Filtering

We can also filter the groups. Let's say we want to compute a mean per group, but we don't want to
include all values from that group, and we also don't want to actually filter the rows from the
dataframe because we need those rows for another aggregation.

In the example below we show how this can be done.

!!! note

    Note that we can define Python functions for clarity.
    These functions don't cost us anything because they return Polars expressions, we don't apply a custom function over a series during runtime of the query.
    Of course, you can write functions that return expressions in Rust, too.

{{code_block('user-guide/expressions/aggregation','filter',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:filter"
```

Do the average age values look nonsensical? That's because we are working with historical data that
dates back to the 1800s and we are doing our computations assuming everyone represented in the
dataset is still alive and kicking.

## Nested grouping

The two previous queries could have been done with a nested `group_by`, but that wouldn't have let
us show off some of these features. 😉 To do a nested `group_by`, simply list the columns that will
be used for grouping.

First, we use a nested `group_by` to figure out how many delegates of a state are “Pro” or “Anti”
administration:

{{code_block('user-guide/expressions/aggregation','nested',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:nested"
```

Next, we use a nested `group_by` to compute the average age of delegates per state and per gender:

{{code_block('user-guide/expressions/aggregation','filter-nested',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:filter-nested"
```

Note that we get the same results but the format of the data is different. Depending on the
situation, one format may be more suitable than the other.

## Sorting

It is common to see a dataframe being sorted for the sole purpose of managing the ordering during a
grouping operation. Let's say that we want to get the names of the oldest and youngest politicians
per state. We could start by sorting and then grouping:

{{code_block('user-guide/expressions/aggregation','sort',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort"
```

However, if we also want to sort the names alphabetically, we need to perform an extra sort
operation. Luckily, we can sort in a `group_by` context without changing the sorting of the
underlying dataframe:

{{code_block('user-guide/expressions/aggregation','sort2',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort2"
```

We can even sort a column with the order induced by another column, and this also works inside the
context `group_by`. This modification to the previous query lets us check if the delegate with the
first name is male or female:

{{code_block('user-guide/expressions/aggregation','sort3',['group_by'])}}

```python exec="on" result="text" session="user-guide/expressions"
--8<-- "python/user-guide/expressions/aggregation.py:sort3"
```

## Do not kill parallelization

!!! warning "Python users only"

    The following section is specific to Python, and doesn't apply to Rust.
    Within Rust, blocks and closures (lambdas) can, and will, be executed concurrently.

Python is generally slower than Rust. Besides the overhead of running “slow” bytecode, Python has to
remain within the constraints of the Global Interpreter Lock (GIL). This means that if you were to
use a `lambda` or a custom Python function to apply during a parallelized phase, Polars' speed is
capped running Python code, preventing any multiple threads from executing the function.

Polars will try to parallelize the computation of the aggregating functions over the groups, so it
is recommended that you avoid using `lambda`s and custom Python functions as much as possible.
Instead, try to stay within the realm of the Polars expression API. This is not always possible,
though, so if you want to learn more about using `lambda`s you can go
[the user guide section on using user-defined functions](user-defined-python-functions.md).
