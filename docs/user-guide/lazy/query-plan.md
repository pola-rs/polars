# Query plan

For any lazy query Polars has both:

- a non-optimized plan with the set of steps code as we provided it and
- an optimized plan with changes made by the query optimizer

We can understand both the non-optimized and optimized query plans with visualization and by printing them as text.

<div style="display:none">
```python exec="on" result="text" session="user-guide/lazy/query-plan"
--8<-- "python/user-guide/lazy/query-plan.py:setup"
```
</div>

Below we consider the following query:

{{code_block('user-guide/lazy/query-plan','plan',[])}}

```python exec="on" session="user-guide/lazy/query-plan"
--8<-- "python/user-guide/lazy/query-plan.py:plan"
```

## Non-optimized query plan

### Graphviz visualization

First we visualise the non-optimized plan by setting `optimized=False`.

{{code_block('user-guide/lazy/query-plan','showplan',['show_graph'])}}

```python exec="on" session="user-guide/lazy/query-plan"
--8<-- "python/user-guide/lazy/query-plan.py:createplan"
```

The query plan visualization should be read from bottom to top. In the visualization:

- each box corresponds to a stage in the query plan
- the `sigma` stands for `SELECTION` and indicates any filter conditions
- the `pi` stands for `PROJECTION` and indicates choosing a subset of columns

### Printed query plan

We can also print the non-optimized plan with `explain(optimized=False)`

{{code_block('user-guide/lazy/query-plan','describe',['explain'])}}

```python exec="on" session="user-guide/lazy/query-plan"
--8<-- "python/user-guide/lazy/query-plan.py:describe"
```

```text
FILTER [(col("comment_karma")) > (0)] FROM WITH_COLUMNS:
 [col("name").str.uppercase()]

    CSV SCAN data/reddit.csv
    PROJECT */6 COLUMNS
```

The printed plan should also be read from bottom to top. This non-optimized plan is roughly equal to:

- read from the `data/reddit.csv` file
- read all 6 columns (where the * wildcard in PROJECT \*/6 COLUMNS means take all columns)
- transform the `name` column to uppercase
- apply a filter on the `comment_karma` column

## Optimized query plan

Now we visualize the optimized plan with `show_graph`.

{{code_block('user-guide/lazy/query-plan','show',['show_graph'])}}

```python exec="on" session="user-guide/lazy/query-plan"
--8<-- "python/user-guide/lazy/query-plan.py:createplan2"
```

We can also print the optimized plan with `explain`

{{code_block('user-guide/lazy/query-plan','optimized',['explain'])}}

```text
 WITH_COLUMNS:
 [col("name").str.uppercase()]

    CSV SCAN data/reddit.csv
    PROJECT */6 COLUMNS
    SELECTION: [(col("comment_karma")) > (0)]
```

The optimized plan is to:

- read the data from the Reddit CSV
- apply the filter on the `comment_karma` column while the CSV is being read line-by-line
- transform the `name` column to uppercase

In this case the query optimizer has identified that the `filter` can be applied while the CSV is read from disk rather than reading the whole file into memory and then applying the filter. This optimization is called _Predicate Pushdown_.
