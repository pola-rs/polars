# Pivots

Pivot a column in a `DataFrame` and perform one of the following aggregations:

- first
- sum
- min
- max
- mean
- median

The pivot operation consists of a group by one, or multiple columns (these will be the
new y-axis), the column that will be pivoted (this will be the new x-axis) and an
aggregation.

## Dataset

{{code_block('user-guide/transformations/pivot','df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:setup"
--8<-- "python/user-guide/transformations/pivot.py:df"
```

## Eager

{{code_block('user-guide/transformations/pivot','eager',['pivot'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:eager"
```

## Lazy

A Polars `LazyFrame` always need to know the schema of a computation statically (before collecting the query).
As a pivot's output schema depends on the data, and it is therefore impossible to determine the schema without
running the query.

Polars could have abstracted this fact for you just like Spark does, but we don't want you to shoot yourself in the foot
with a shotgun. The cost should be clear upfront.

{{code_block('user-guide/transformations/pivot','lazy',['pivot'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:lazy"
```
