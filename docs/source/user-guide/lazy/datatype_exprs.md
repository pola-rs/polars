# DataType Expressions

In your lazy queries, you may want to reason about the datatypes of columns or expressions used in
your queries. DataType expressions allow for the inspection and manipulation of datatypes that are
used in your query. The datatypes are resolved during query planning and behave the same as static
datatypes during runtime.

DataType expressions can be especially useful when you don't have full control over input data. This
can occur when you try to compartmentalize code, write utility functions or are loading data from
heterogeneous data sources. DataType expressions also allow you to express relations between the
datatype of expressions or columns.

## Basic Usage

DataType expressions often start with `pl.dtype_of`. This allows inspecting the datatype of a column
or expression.

<div style="display:none">
```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:setup"
```
</div>

{{code_block('user-guide/lazy/datatype_exprs','basic',['dtype_of'])}}

```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:basic"
```

These expressions can be manipulated in various ways to transform them into the datatype that you
need.

{{code_block('user-guide/lazy/datatype_exprs','basic-manipulation',[])}}

```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:basic-manipulation"
```

You can also inspect information about the datatype to use at runtime.

{{code_block('user-guide/lazy/datatype_exprs','basic-inspect',[])}}

```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:basic-inspect"
```

## Expressing relations between datatypes

Datatypes can help with utility functions by being able to express the relation between the output
datatype of two expressions. The following example allows you to express that `map_batches` has the
same output datatype as input datatype.

{{code_block('user-guide/lazy/datatype_exprs','inspect',['map_batches'])}}

```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:inspect"
```

Similarly, you want to express that one column needs to be casted to the datatype of another column.

{{code_block('user-guide/lazy/datatype_exprs','cast',['cast'])}}

```python exec="on" result="text" session="user-guide/lazy/datatype_exprs"
--8<-- "python/user-guide/lazy/datatype_exprs.py:cast"
```
