# Lazy API

Polars supports two modes of operation: lazy and eager. The examples so far have used the eager API, in which the query is executed immediately.
In the lazy API, the query is only evaluated once it is _collected_. Deferring the execution to the last minute can have significant performance advantages and is why the lazy API is preferred in most cases. Let us demonstrate this with an example:

{{code_block('user-guide/concepts/lazy-vs-eager','eager',['read_csv'])}}

In this example we use the eager API to:

1. Read the iris [dataset](https://archive.ics.uci.edu/dataset/53/iris).
1. Filter the dataset based on sepal length.
1. Calculate the mean of the sepal width per species.

Every step is executed immediately returning the intermediate results. This can be very wasteful as we might do work or load extra data that is not being used. If we instead used the lazy API and waited on execution until all the steps are defined then the query planner could perform various optimizations. In this case:

- Predicate pushdown: Apply filters as early as possible while reading the dataset, thus only reading rows with sepal length greater than 5.
- Projection pushdown: Select only the columns that are needed while reading the dataset, thus removing the need to load additional columns (e.g., petal length and petal width).

{{code_block('user-guide/concepts/lazy-vs-eager','lazy',['scan_csv'])}}

These will significantly lower the load on memory & CPU thus allowing you to fit bigger datasets in memory and process them faster. Once the query is defined you call `collect` to inform Polars that you want to execute it. You can [learn more about the lazy API in its dedicated chapter](../lazy/index.md).

!!! info "Eager API"

    In many cases the eager API is actually calling the lazy API under the hood and immediately collecting the result. This has the benefit that within the query itself optimization(s) made by the query planner can still take place.

## When to use which

In general, the lazy API should be preferred unless you are either interested in the intermediate results or are doing exploratory work and don't know yet what your query is going to look like.

## Previewing the query plan

When using the lazy API you can use the function `explain` to ask Polars to create a description of the query plan that will be executed once you collect the results.
This can be useful if you want to see what types of optimizations Polars performs on your queries.
We can ask Polars to explain the query `q` we defined above:

{{code_block('user-guide/concepts/lazy-vs-eager','explain',['explain'])}}

```python exec="on" result="text" session="user-guide/concepts/lazy-api"
--8<-- "python/user-guide/concepts/lazy-vs-eager.py:import"
--8<-- "python/user-guide/concepts/lazy-vs-eager.py:lazy"
--8<-- "python/user-guide/concepts/lazy-vs-eager.py:explain"
```

Immediately, we can see in the explanation that Polars did apply predicate pushdown, as it is only reading rows where the sepal length is greater than 5, and it did apply projection pushdown, as it is only reading the columns that are needed by the query.

The function `explain` can also be used to see how expression expansion will unfold in the context of a given schema.
Consider the example expression from the [section on expression expansion](expressions-and-contexts.md#expression-expansion):

```python
(pl.col(pl.Float64) * 1.1).name.suffix("*1.1")
```

We can use `explain` to see how this expression would evaluate against an arbitrary schema:

=== ":fontawesome-brands-python: Python"
[:material-api: `explain`](https://docs.pola.rs/api/python/stable/reference/lazyframe/api/polars.LazyFrame.explain.html)

```python
--8<-- "python/user-guide/concepts/lazy-vs-eager.py:explain-expression-expansion"
```

```python exec="on" result="text" session="user-guide/concepts/lazy-api"
--8<-- "python/user-guide/concepts/lazy-vs-eager.py:explain-expression-expansion"
```
