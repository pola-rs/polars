# Expressions and contexts

Polars has developed its own Domain Specific Language (DSL) for transforming data. The language is
very easy to use and allows for complex queries that remain human readable. Expressions and
contexts, which will be introduced here, are very important in achieving this readability while also
allowing the Polars query engine to optimize your queries to make them run as fast as possible.

## Expressions

In Polars, an _expression_ is a lazy representation of a data transformation. Expressions are
modular and flexible, which means you can use them as building blocks to build more complex
expressions. Here is an example of a Polars expression:

```python
--8<-- "python/user-guide/concepts/expressions.py:expression"
```

As you might be able to guess, this expression takes a column named “weight” and divides its values
by the square of the values in a column “height”, computing a person's BMI.

The code above expresses an abstract computation that we can save in a variable, manipulate further,
or just print:

```python
--8<-- "python/user-guide/concepts/expressions.py:print-expr"
```

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:expression"
--8<-- "python/user-guide/concepts/expressions.py:print-expr"
```

Because expressions are lazy, no computations have taken place yet. That's what we need contexts
for.

## Contexts

Polars expressions need a _context_ in which they are executed to produce a result. Depending on the
context it is used in, the same Polars expression can produce different results. In this section, we
will learn about the four most common contexts that Polars provides[^1]:

1. `select`
2. `with_columns`
3. `filter`
4. `group_by`

We use the dataframe below to show how each of the contexts works.

{{code_block('user-guide/concepts/expressions','df',[])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:df"
```

### `select`

The selection context `select` applies expressions over columns. The context `select` may produce
new columns that are aggregations, combinations of other columns, or literals:

{{code_block('user-guide/concepts/expressions','select-1',['select'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:select-1"
```

The expressions in a context `select` must produce series that are all the same length or they must
produce a scalar. Scalars will be broadcast to match the length of the remaining series. Literals,
like the number used above, are also broadcast.

Note that broadcasting can also occur within expressions. For instance, consider the expression
below:

{{code_block('user-guide/concepts/expressions','select-2',['select'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:select-2"
```

Both the subtraction and the division use broadcasting within the expression because the
subexpressions that compute the mean and the standard deviation evaluate to single values.

The context `select` is very flexible and powerful and allows you to evaluate arbitrary expressions
independent of, and in parallel to, each other. This is also true of the other contexts that we will
see next.

### `with_columns`

The context `with_columns` is very similar to the context `select`. The main difference between the
two is that the context `with_columns` creates a new dataframe that contains the columns from the
original dataframe and the new columns according to its input expressions, whereas the context
`select` only includes the columns selected by its input expressions:

{{code_block('user-guide/concepts/expressions','with_columns-1',['with_columns'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:with_columns-1"
```

Because of this difference between `select` and `with_columns`, the expressions used in a context
`with_columns` must produce series that have the same length as the original columns in the
dataframe, whereas it is enough for the expressions in the context `select` to produce series that
have the same length among them.

### `filter`

The context `filter` filters the rows of a dataframe based on one or more expressions that evaluate
to the Boolean data type.

{{code_block('user-guide/concepts/expressions','filter-1',['filter'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:filter-1"
```

### `group_by` and aggregations

In the context `group_by`, rows are grouped according to the unique values of the grouping
expressions. You can then apply expressions to the resulting groups, which may be of variable
lengths.

When using the context `group_by`, you can use an expression to compute the groupings dynamically:

{{code_block('user-guide/concepts/expressions','group_by-1',['group_by'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:group_by-1"
```

After using `group_by` we use `agg` to apply aggregating expressions to the groups. Since in the
example above we only specified the name of a column, we get the groups of that column as lists.

We can specify as many grouping expressions as we'd like and the context `group_by` will group the
rows according to the distinct values across the expressions specified. Here, we group by a
combination of decade of birth and whether the person is shorter than 1.7 metres:

{{code_block('user-guide/concepts/expressions','group_by-2',['group_by'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:group_by-2"
```

The resulting dataframe, after applying aggregating expressions, contains one column per each
grouping expression on the left and then as many columns as needed to represent the results of the
aggregating expressions. In turn, we can specify as many aggregating expressions as we want:

{{code_block('user-guide/concepts/expressions','group_by-3',['group_by'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:group_by-3"
```

See also `group_by_dynamic` and `rolling` for other grouping contexts.

## Expression expansion

The last example contained two grouping expressions and three aggregating expressions, and yet the
resulting dataframe contained six columns instead of five. If we look closely, the last aggregating
expression mentioned two different columns: “weight” and “height”.

Polars expressions support a feature called _expression expansion_. Expression expansion is like a
shorthand notation for when you want to apply the same transformation to multiple columns. As we
have seen, the expression

```python
pl.col("weight", "height").mean().name.prefix("avg_")
```

will compute the mean value of the columns “weight” and “height” and will rename them as
“avg_weight” and “avg_height”, respectively. In fact, the expression above is equivalent to using
the two following expressions:

```python
[
    pl.col("weight").mean().alias("avg_weight"),
    pl.col("height").mean().alias("avg_height"),
]
```

In this case, this expression expands into two independent expressions that Polars can execute in
parallel. In other cases, we may not be able to know in advance how many independent expressions an
expression will unfold into.

Consider this simple but elucidative example:

```python
(pl.col(pl.Float64) * 1.1).name.suffix("*1.1")
```

This expression will multiply all columns with data type `Float64` by `1.1`. The number of columns
this applies to depends on the schema of each dataframe. In the case of the dataframe we have been
using, it applies to two columns:

{{code_block('user-guide/concepts/expressions','expression-expansion-1',['group_by'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:expression-expansion-1"
```

In the case of the dataframe `df2` below, the same expression expands to 0 columns because no column
has the data type `Float64`:

{{code_block('user-guide/concepts/expressions','expression-expansion-2',['group_by'])}}

```python exec="on" result="text" session="user-guide/concepts/expressions-and-contexts"
--8<-- "python/user-guide/concepts/expressions.py:expression-expansion-2"
```

It is equally easy to imagine a scenario where the same expression would expand to dozens of
columns.

Next, you will learn about
[the lazy API and the function `explain`](lazy-api.md#previewing-the-query-plan), which you can use
to preview what an expression will expand to given a schema.

## Conclusion

Because expressions are lazy, when you use an expression inside a context Polars can try to simplify
your expression before running the data transformation it expresses. Separate expressions within a
context are embarrassingly parallel and Polars will take advantage of that, while also parallelizing
expression execution when using expression expansion. Further performance gains can be obtained when
using [the lazy API of Polars](lazy-api.md), which is introduced next.

We have only scratched the surface of the capabilities of expressions. There are a ton more
expressions and they can be combined in a variety of ways. See the
[section on expressions](../expressions/index.md) for a deeper dive on the different types of
expressions available.

[^1]: There are additional List and SQL contexts which are covered later in this guide. But for
simplicity, we leave them out of scope for now.
