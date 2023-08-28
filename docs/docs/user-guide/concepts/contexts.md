# Contexts

Polars has developed its own Domain Specific Language (DSL) for transforming data. The language is very easy to use and allows for complex queries that remain human readable. The two core components of the language are Contexts and Expressions, the latter we will cover in the next section. 

A context, as implied by the name, refers to the context in which an expression needs to be evaluated. There are three main contexts [^1]: 

1. Selection: `df.select([..])`, `df.with_columns([..])`
1. Filtering: `df.filter()`
1. Groupby / Aggregation: `df.groupby(..).agg([..])`

The examples below are performed on the following `DataFrame`:

{{code_block('user-guide/concepts/contexts','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:setup"
--8<-- "python/user-guide/concepts/contexts.py:dataframe"
```

## Select 

In the `select` context the selection applies expressions over columns. The expressions in this context must produce `Series` that are all the same length or have a length of 1.

A `Series` of a length of 1 will be broadcasted to match the height of the `DataFrame`. Note that a select may produce new columns that are aggregations, combinations of expressions, or literals.

{{code_block('user-guide/concepts/contexts','select',['select'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:select"
```

As you can see from the query the `select` context is very powerful and allows you to perform arbitrary expressions independent (and in parallel) of each other. 

Similarly to the `select` statement there is the `with_columns` statement which also is an entrance to the selection context. The main difference is that `with_columns` retains the original columns and adds new ones while `select` drops the original columns.

{{code_block('user-guide/concepts/contexts','with_columns',['with_columns'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:with_columns"
```

## Filter 

In the `filter` context you filter the existing dataframe based on arbritary expression which evaluates to the `Boolean` data type. 

{{code_block('user-guide/concepts/contexts','filter',['filter'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:filter"
```

## Groupby / Aggregation 

In the `groupby` context expressions work on groups and thus may yield results of any length (a group may have many members).

{{code_block('user-guide/concepts/contexts','groupby',['groupby'])}}

```python exec="on" result="text" session="user-guide/contexts"
--8<-- "python/user-guide/concepts/contexts.py:groupby"
```

As you can see from the result all expressions are applied to the group defined by the `groupby` context. Besides the standard `groupby`, `groupby_dynamic`, and `groupby_rolling` are also entrances to the groupby context.

[^1]: There are additional List and SQL contexts which are covered later in this guide. But for simplicity, we leave them out of scope for now. 
