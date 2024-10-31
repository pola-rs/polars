# Coming from Apache Spark

## Column-based API vs. Row-based API

Whereas the `Spark` `DataFrame` is analogous to a collection of rows, a Polars `DataFrame` is closer to a collection of columns. This means that you can combine columns in Polars in ways that are not possible in `Spark`, because `Spark` preserves the relationship of the data in each row.

Consider this sample dataset:

```python
import polars as pl

df = pl.DataFrame({
    "foo": ["a", "b", "c", "d", "d"],
    "bar": [1, 2, 3, 4, 5],
})

dfs = spark.createDataFrame(
    [
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("d", 4),
        ("d", 5),
    ],
    schema=["foo", "bar"],
)
```

### Example 1: Combining `head` and `sum`

In Polars you can write something like this:

```python
df.select(
    pl.col("foo").sort().head(2),
    pl.col("bar").filter(pl.col("foo") == "d").sum()
)
```

Output:

```
shape: (2, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ str ┆ i64 │
╞═════╪═════╡
│ a   ┆ 9   │
├╌╌╌╌╌┼╌╌╌╌╌┤
│ b   ┆ 9   │
└─────┴─────┘
```

The expressions on columns `foo` and `bar` are completely independent. Since the expression on `bar` returns a single value, that value is repeated for each value output by the expression on `foo`. But `a` and `b` have no relation to the data that produced the sum of `9`.

To do something similar in `Spark`, you'd need to compute the sum separately and provide it as a literal:

```python
from pyspark.sql.functions import col, sum, lit

bar_sum = (
    dfs
    .where(col("foo") == "d")
    .groupBy()
    .agg(sum(col("bar")))
    .take(1)[0][0]
)

(
    dfs
    .orderBy("foo")
    .limit(2)
    .withColumn("bar", lit(bar_sum))
    .show()
)
```

Output:

```
+---+---+
|foo|bar|
+---+---+
|  a|  9|
|  b|  9|
+---+---+
```

### Example 2: Combining Two `head`s

In Polars you can combine two different `head` expressions on the same DataFrame, provided that they return the same number of values.

```python
df.select(
    pl.col("foo").sort().head(2),
    pl.col("bar").sort(descending=True).head(2),
)
```

Output:

```
shape: (3, 2)
┌─────┬─────┐
│ foo ┆ bar │
│ --- ┆ --- │
│ str ┆ i64 │
╞═════╪═════╡
│ a   ┆ 5   │
├╌╌╌╌╌┼╌╌╌╌╌┤
│ b   ┆ 4   │
└─────┴─────┘
```

Again, the two `head` expressions here are completely independent, and the pairing of `a` to `5` and `b` to `4` results purely from the juxtaposition of the two columns output by the expressions.

To accomplish something similar in `Spark`, you would need to generate an artificial key that enables you to join the values in this way.

```python
from pyspark.sql import Window
from pyspark.sql.functions import row_number

foo_dfs = (
    dfs
    .withColumn(
        "rownum",
        row_number().over(Window.orderBy("foo"))
    )
)

bar_dfs = (
    dfs
    .withColumn(
        "rownum",
        row_number().over(Window.orderBy(col("bar").desc()))
    )
)

(
    foo_dfs.alias("foo")
    .join(bar_dfs.alias("bar"), on="rownum")
    .select("foo.foo", "bar.bar")
    .limit(2)
    .show()
)
```

Output:

```
+---+---+
|foo|bar|
+---+---+
|  a|  5|
|  b|  4|
+---+---+
```
