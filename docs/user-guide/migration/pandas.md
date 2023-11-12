# Coming from Pandas

Here we set out the key points that anyone who has experience with pandas and wants to
try Polars should know. We include both differences in the concepts the libraries are
built on and differences in how you should write Polars code compared to pandas
code.

## Differences in concepts between Polars and pandas

### Polars does not have a multi-index/index

pandas gives a label to each row with an index. Polars does not use an index and
each row is indexed by its integer position in the table.

Polars aims to have predictable results and readable queries, as such we think an index does not help us reach that
objective. We believe the semantics of a query should not change by the state of an index or a `reset_index` call.

In Polars a DataFrame will always be a 2D table with heterogeneous data-types. The data-types may have nesting, but the
table itself will not.
Operations like resampling will be done by specialized functions or methods that act like 'verbs' on a table explicitly
stating the columns that that 'verb' operates on. As such, it is our conviction that not having indices make things simpler,
more explicit, more readable and less error-prone.

Note that an 'index' data structure as known in databases will be used by Polars as an optimization technique.

### Polars uses Apache Arrow arrays to represent data in memory while pandas uses NumPy arrays

Polars represents data in memory with Arrow arrays while pandas represents data in
memory with NumPy arrays. Apache Arrow is an emerging standard for in-memory columnar
analytics that can accelerate data load times, reduce memory usage and accelerate
calculations.

Polars can convert data to NumPy format with the `to_numpy` method.

### Polars has more support for parallel operations than pandas

Polars exploits the strong support for concurrency in Rust to run many operations in
parallel. While some operations in pandas are multi-threaded the core of the library
is single-threaded and an additional library such as `Dask` must be used to parallelize
operations.

### Polars can lazily evaluate queries and apply query optimization

Eager evaluation is when code is evaluated as soon as you run the code. Lazy evaluation
is when running a line of code means that the underlying logic is added to a query plan
rather than being evaluated.

Polars supports eager evaluation and lazy evaluation whereas pandas only supports
eager evaluation. The lazy evaluation mode is powerful because Polars carries out
automatic query optimization when it examines the query plan and looks for ways to
accelerate the query or reduce memory usage.

`Dask` also supports lazy evaluation when it generates a query plan. However, `Dask`
does not carry out query optimization on the query plan.

## Key syntax differences

Users coming from pandas generally need to know one thing...

```
polars != pandas
```

If your Polars code looks like it could be pandas code, it might run, but it likely
runs slower than it should.

Let's go through some typical pandas code and see how we might rewrite it in Polars.

### Selecting data

As there is no index in Polars there is no `.loc` or `iloc` method in Polars - and
there is also no `SettingWithCopyWarning` in Polars.

However, the best way to select data in Polars is to use the expression API. For
example, if you want to select a column in pandas you can do one of the following:

```python
df['a']
df.loc[:,'a']
```

but in Polars you would use the `.select` method:

```python
df.select('a')
```

If you want to select rows based on the values then in Polars you use the `.filter`
method:

```python
df.filter(pl.col('a') < 10)
```

As noted in the section on expressions below, Polars can run operations in `.select`
and `filter` in parallel and Polars can carry out query optimization on the full set
of data selection criteria.

### Be lazy

Working in lazy evaluation mode is straightforward and should be your default in
Polars as the lazy mode allows Polars to do query optimization.

We can run in lazy mode by either using an implicitly lazy function (such as `scan_csv`)
or explicitly using the `lazy` method.

Take the following simple example where we read a CSV file from disk and do a group by.
The CSV file has numerous columns but we just want to do a group by on one of the id
columns (`id1`) and then sum by a value column (`v1`). In pandas this would be:

```python
df = pd.read_csv(csv_file, usecols=['id1','v1'])
grouped_df = df.loc[:,['id1','v1']].groupby('id1').sum('v1')
```

In Polars you can build this query in lazy mode with query optimization and evaluate
it by replacing the eager pandas function `read_csv` with the implicitly lazy Polars
function `scan_csv`:

```python
df = pl.scan_csv(csv_file)
grouped_df = df.group_by('id1').agg(pl.col('v1').sum()).collect()
```

Polars optimizes this query by identifying that only the `id1` and `v1` columns are
relevant and so will only read these columns from the CSV. By calling the `.collect`
method at the end of the second line we instruct Polars to eagerly evaluate the query.

If you do want to run this query in eager mode you can just replace `scan_csv` with
`read_csv` in the Polars code.

Read more about working with lazy evaluation in the
[lazy API](../lazy/using.md) section.

### Express yourself

A typical pandas script consists of multiple data transformations that are executed
sequentially. However, in Polars these transformations can be executed in parallel
using expressions.

#### Column assignment

We have a dataframe `df` with a column called `value`. We want to add two new columns, a
column called `tenXValue` where the `value` column is multiplied by 10 and a column
called `hundredXValue` where the `value` column is multiplied by 100.

In pandas this would be:

```python
df.assign(
    tenXValue=lambda df_: df_.value * 10,
    hundredXValue=lambda df_: df_.value * 100
)
```

These column assignments are executed sequentially.

In Polars we add columns to `df` using the `.with_columns` method:

```python
df.with_columns(
    tenXValue=pl.col("value") * 10,
    hundredXValue=pl.col("value") * 100,
)
```

These column assignments are executed in parallel.

#### Column assignment based on predicate

In this case we have a dataframe `df` with columns `a`,`b` and `c`. We want to re-assign
the values in column `a` based on a condition. When the value in column `c` is equal to
2 then we replace the value in `a` with the value in `b`.

In pandas this would be:

```python
df.assign(a=lambda df_: df_.a.where(df_.c != 2, df_.b))
```

while in Polars this would be:

```python
df.with_columns(
    pl.when(pl.col("c") == 2)
    .then(pl.col("b"))
    .otherwise(pl.col("a")).alias("a")
)
```

Polars can compute every branch of an `if -> then -> otherwise` in
parallel. This is valuable, when the branches get more expensive to compute.

#### Filtering

We want to filter the dataframe `df` with housing data based on some criteria.

In pandas you filter the dataframe by passing Boolean expressions to the `query` method:

```python
df.query('m2_living > 2500 and price < 300000')
```

while in Polars you call the `filter` method:

```python
df.filter(
    (pl.col("m2_living") > 2500) & (pl.col("price") < 300000)
)
```

The query optimizer in Polars can also detect if you write multiple filters separately
and combine them into a single filter in the optimized plan.

## pandas transform

The pandas documentation demonstrates an operation on a group by called `transform`. In
this case we have a dataframe `df` and we want a new column showing the number of rows
in each group.

In pandas we have:

```python
df = pd.DataFrame({
    "c": [1, 1, 1, 2, 2, 2, 2],
    "type": ["m", "n", "o", "m", "m", "n", "n"],
})

df["size"] = df.groupby("c")["type"].transform(len)
```

Here pandas does a group by on `"c"`, takes column `"type"`, computes the group length
and then joins the result back to the original `DataFrame` producing:

```
   c type size
0  1    m    3
1  1    n    3
2  1    o    3
3  2    m    4
4  2    m    4
5  2    n    4
6  2    n    4
```

In Polars the same can be achieved with `window` functions:

```python
df.select(
    pl.all(),
    pl.col("type").count().over("c").alias("size")
)
```

```
shape: (7, 3)
┌─────┬──────┬──────┐
│ c   ┆ type ┆ size │
│ --- ┆ ---  ┆ ---  │
│ i64 ┆ str  ┆ u32  │
╞═════╪══════╪══════╡
│ 1   ┆ m    ┆ 3    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 1   ┆ n    ┆ 3    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 1   ┆ o    ┆ 3    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2   ┆ m    ┆ 4    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2   ┆ m    ┆ 4    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2   ┆ n    ┆ 4    │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┤
│ 2   ┆ n    ┆ 4    │
└─────┴──────┴──────┘
```

Because we can store the whole operation in a single expression, we can combine several
`window` functions and even combine different groups!

Polars will cache window expressions that are applied over the same group, so storing
them in a single `select` is both convenient **and** optimal. In the following example
we look at a case where we are calculating group statistics over `"c"` twice:

```python
df.select(
    pl.all(),
    pl.col("c").count().over("c").alias("size"),
    pl.col("c").sum().over("type").alias("sum"),
    pl.col("c").reverse().over("c").flatten().alias("reverse_type")
)
```

```
shape: (7, 5)
┌─────┬──────┬──────┬─────┬──────────────┐
│ c   ┆ type ┆ size ┆ sum ┆ reverse_type │
│ --- ┆ ---  ┆ ---  ┆ --- ┆ ---          │
│ i64 ┆ str  ┆ u32  ┆ i64 ┆ i64          │
╞═════╪══════╪══════╪═════╪══════════════╡
│ 1   ┆ m    ┆ 3    ┆ 5   ┆ 2            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 1   ┆ n    ┆ 3    ┆ 5   ┆ 2            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 1   ┆ o    ┆ 3    ┆ 1   ┆ 2            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 2   ┆ m    ┆ 4    ┆ 5   ┆ 2            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 2   ┆ m    ┆ 4    ┆ 5   ┆ 1            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 2   ┆ n    ┆ 4    ┆ 5   ┆ 1            │
├╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ 2   ┆ n    ┆ 4    ┆ 5   ┆ 1            │
└─────┴──────┴──────┴─────┴──────────────┘
```

## Missing data

pandas uses `NaN` and/or `None` values to indicate missing values depending on the dtype of the column. In addition the behaviour in pandas varies depending on whether the default dtypes or optional nullable arrays are used. In Polars missing data corresponds to a `null` value for all data types.

For float columns Polars permits the use of `NaN` values. These `NaN` values are not considered to be missing data but instead a special floating point value.

In pandas an integer column with missing values is cast to be a float column with `NaN` values for the missing values (unless using optional nullable integer dtypes). In Polars any missing values in an integer column are simply `null` values and the column remains an integer column.

See the [missing data](../expressions/null.md) section for more details.
