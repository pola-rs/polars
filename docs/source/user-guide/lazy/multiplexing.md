# Multiplexing queries

<div style="display:none">
```python exec="on" result="text" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:setup"
```
</div>

In the [Sources and Sinks](./sources_sinks.md) page, we already discussed multiplexing as a way to
split a query into multiple sinks. This page will go a bit deeper in this concept, as it is
important to understand when combining `LazyFrame`s with procedural programming constructs.

When dealing with eager dataframes, it is very common to keep state in a temporary variable. Let's
look at the following example. Below we create a `DataFrame` with 10 unique elements in a random
order (so that Polars doesn't hit any fast paths for sorted keys).

{{code_block('user-guide/lazy/multiplexing','dataframe',[])}}

```python exec="on" result="text" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:dataframe"
```

## Eager

If you deal with the Polars eager API, making a variable and iterating over that temporary
`DataFrame` gives the result you expect, as the result of the group-by is stored in `df1`. Even
though the output order is unstable, it doesn't matter as it is eagerly evaluated. The follow
snippet therefore doesn't raise and the assert passes.
{{code_block('user-guide/lazy/multiplexing','eager',[])}}

## Lazy

Now if we tried this naively with `LazyFrame`s, this would fail.

{{code_block('user-guide/lazy/multiplexing','lazy',[])}}

```python
AssertionError: DataFrames are different (value mismatch for column 'n')
[left]:  [9, 2, 0, 5, 3, 1, 7, 8, 6, 4]
[right]: [0, 9, 6, 8, 2, 5, 4, 3, 1, 7]
```

The reason this fails is that `lf1` doesn't contain the materialized result of
`df.lazy().group_by("n").len()`, it instead holds the query plan in that variable.

```python exec="on" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:plan_0"
```

This means that every time we branch of this `LazyFrame` and call `collect` we re-evaluate the
group-by. Besides being expensive, this also leads to unexpected results if you assume that the
output is stable (which isn't the case here).

In the example above you are actually evaluating 2 query plans:

**Plan 1**

```python exec="on" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:plan_1"
```

**Plan 2**

```python exec="on" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:plan_2"
```

## Combine the query plans

To circumvent this, we must give Polars the opportunity to look at all the query plans in a single
optimization and execution pass. This can be done by passing the diverging `LazyFrame`'s to the
`collect_all` function.

{{code_block('user-guide/lazy/multiplexing','collect_all',[])}}

If we explain the combined queries with `pl.explain_all`, we can also observe that they are shared
under a single "SINK_MULTIPLE" evaluation and that the optimizer has recognized that parts of the
query come from the same subplan, indicated by the inserted "CACHE" nodes.

```python exec="on" result="text" session="user-guide/lazy/multiplexing"
--8<-- "python/user-guide/lazy/multiplexing.py:explain_all"
```

Combining related subplans in a single execution unit with `pl.collect_all` can thus lead to large
performance increases and allows diverging query plans, storing temporary tables, and a more
procedural programming style.
