# Contexts

Polars has developed its own Domain Specific Language (DSL) for transforming data. The language is very easy to use and allows for complex queries that remain human readable. The two core components of the language are Contexts and Expressions, the latter we will cover in the next section.

A context, as implied by the name, refers to the context in which an expression needs to be evaluated. There are three main contexts [^1]:

1. Selection: `df.select([..])`, `df.with_columns([..])`
1. Filtering: `df.filter()`
1. Group by / Aggregation: `df.group_by(..).agg([..])`

The examples below are performed on the following `DataFrame`:

{{code_block('user-guide/concepts/contexts','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:setup"
--8<-- "python/user-guide/concepts/contexts.py:dataframe"
```

## Selection

There are two selection contexts: `select` and `with_columns`.

The `select` context applies expressions over columns. A `select` may produce new columns that are aggregations, combinations of expressions, or literals. 

The expressions in a `select` context must produce `Series` that are all the same length or have a length of 1. Literals are treated as length-1 `Series`. 

When some expressions produce length-1 `Series` and some do not, the length-1 `Series` will be broadcast to match the length of the remaining `Series`. Note that broadcasting can also occur within expressions: for instance, in `pl.col.value() / pl.col.value.sum()`, each element of the `value` column is divided by the column's sum.  

{{code_block('user-guide/concepts/contexts','select',['select'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:select"
```

As you can see from the query the `select` context is very powerful and allows you to evaluate arbitrary expressions independent of (and in parallel to) each other.

The other selection context is `with_columns`. The main difference between `select` and `with_columns` is that `with_columns` retains the original columns and adds new ones, whereas `select` drops the original columns. The other difference is that `with_columns` always yields a `DataFrame` of the same length as the original, so broadcasting will occur even if every `Series` produced has a length of 1. For instance, `df.select(pl.all().sum())` results in a length-1 DataFrame, whereas `df.with_columns(pl.all().sum())` broadcasts each of the sums to the length of the original `DataFrame`.

{{code_block('user-guide/concepts/contexts','with_columns',['with_columns'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:with_columns"
```

## Filtering

The `filter` context filters a `DataFrame` based on one or more expressions that evaluate to the `Boolean` data type.

{{code_block('user-guide/concepts/contexts','filter',['filter'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:filter"
```

## Group by / aggregation

In the `group_by` context, expressions work on groups and thus may yield results of any length (a group may have many members).

{{code_block('user-guide/concepts/contexts','group_by',['group_by'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:group_by"
```

As you can see from the result all expressions are applied to the group defined by the `group_by` context. Besides the standard `group_by`, `group_by_dynamic`, and `group_by_rolling` are also entrances to the group by context.

[^1]: There are additional List and SQL contexts which are covered later in this guide. But for simplicity, we leave them out of scope for now.
