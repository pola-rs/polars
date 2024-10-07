# Data types and structures

## Data types

Polars supports a variety of data types that fall broadly under the following categories:

- Numeric data types: signed integers, unsigned integers, floating point numbers, and decimals.
- Nested data types: lists, structs, and arrays.
- Temporal: dates, datetimes, times, and time deltas.
- Miscellaneous: strings, binary data, Booleans, categoricals, enums, and objects.

All types support missing values represented by the special value `null`.
This is not to be conflated with the special value `NaN` in floating number data types; see the [section about floating point numbers](#floating-point-numbers) for more information.

You can also find a [full table with all data types supported in the appendix](#appendix-full-data-types-table) with notes on when to use each data type and with links to relevant parts of the documentation.

## Series

The core base data structures provided by Polars are series and dataframes.
A series is a 1-dimensional homogeneous data structure.
By “homogeneous” we mean that all elements inside a series have the same data type.
The snippet below shows how to create a named series:

{{code_block('user-guide/concepts/data-types-and-structures','series',['Series'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:series"
```

When creating a series, Polars will infer the data type from the values you provide.
You can specify a concrete data type to override the inference mechanism:

{{code_block('user-guide/concepts/data-types-and-structures','series-dtype',['Series'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:series-dtype"
```

## Dataframe

A dataframe is a 2-dimensional heterogeneous data structure that contains uniquely named series.
By holding your data in a dataframe you will be able to use the Polars API to write queries that manipulate your data.
You will be able to do this by using the [contexts and expressions provided by Polars](expressions-and-contexts.md) that we will talk about next.

The snippet below shows how to create a dataframe from a dictionary of lists:

{{code_block('user-guide/concepts/data-types-and-structures','df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:df"
```

### Inspecting a dataframe

In this subsection we will show some useful methods to quickly inspect a dataframe.
We will use the dataframe we created earlier as a starting point.

#### Head

The function `head` shows the first rows of a dataframe.
By default, you get the first 5 rows but you can also specify the number of rows you want:

{{code_block('user-guide/concepts/data-types-and-structures','head',['head'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:head"
```

#### Glimpse

The function `glimpse` is another function that shows the values of the first few rows of a dataframe, but formats the output differently from `head`.
Here, each line of the output corresponds to a single column, making it easier to take inspect wider dataframes:

=== ":fontawesome-brands-python: Python"
[:material-api: `glimpse`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.glimpse.html)

```python
--8<-- "python/user-guide/concepts/data-types-and-structures.py:glimpse"
```

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:glimpse"
```

!!! info
`glimpse` is only available for Python users.

#### Tail

The function `tail` shows the last rows of a dataframe.
By default, you get the last 5 rows but you can also specify the number of rows you want, similar to how `head` works:

{{code_block('user-guide/concepts/data-types-and-structures','tail',['tail'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:tail"
```

#### Sample

If you think the first or last rows of your dataframe are not representative of your data, you can use `sample` to get an arbitrary number of randomly selected rows from the DataFrame.
Note that the rows are not necessarily returned in the same order as they appear in the dataframe:

{{code_block('user-guide/concepts/data-types-and-structures','sample',['sample'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:sample"
```

#### Describe

You can also use `describe` to compute summary statistics for all columns of your dataframe:

{{code_block('user-guide/concepts/data-types-and-structures','describe',['describe'])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:describe"
```

## Schema

When talking about data (in a dataframe or otherwise) we can refer to its schema.
The schema is a mapping of column or series names to the data types of those same columns or series.

Much like with series, Polars will infer the schema of a dataframe when you create it but you can override the inference system if needed.
You can check the schema of a dataframe with `schema`:

{{code_block('user-guide/concepts/data-types-and-structures','schema',[])}}

```python exec="on" result="text" session="user-guide/data-types-and-structures"
--8<-- "python/user-guide/concepts/data-types-and-structures.py:schema"
```

## Data types internals

Polars utilizes the [Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) for its data orientation.
Following this specification allows Polars to transfer data to/from other tools that also use the Arrow specification with little to no overhead.

Polars gets most of its performance from its query engine, the optimizations it performs on your query plans, and from the parallelization that it employs when running [your expressions](expressions-and-contexts.md#expressions).

## Floating point numbers

Polars generally follows the IEEE 754 floating point standard for `Float32` and `Float64`, with some exceptions:

- Any `NaN` compares equal to any other `NaN`, and greater than any non-`NaN` value.
- Operations do not guarantee any particular behavior on the sign of zero or `NaN`,
  nor on the payload of `NaN` values. This is not just limited to arithmetic operations,
  e.g. a sort or group by operation may canonicalize all zeroes to +0 and all `NaN`s
  to a positive `NaN` without payload for efficient equality checks.

Polars always attempts to provide reasonably accurate results for floating point computations but does not provide guarantees
on the error unless mentioned otherwise. Generally speaking 100% accurate results are infeasibly expensive to achieve (requiring
much larger internal representations than 64-bit floats), and thus some error is always to be expected.

## Appendix: full data types table

| Type(s)                               | Details                                                                                                                                                                                                                                                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Boolean`                             | Boolean type that is bit packed efficiently.                                                                                                                                                                                                                                                                             |
| `Int8`, `Int16`, `Int32`, `Int64`     | Varying-precision signed integer types.                                                                                                                                                                                                                                                                                  |
| `UInt8`, `UInt16`, `UInt32`, `UInt64` | Varying-precision unsigned integer types.                                                                                                                                                                                                                                                                                |
| `Float32`, `Float64`                  | Varying-precision signed floating point numbers.                                                                                                                                                                                                                                                                         |
| `Decimal`                             | Decimal 128-bit type with optional precision and non-negative scale. Use this if you need fine-grained control over the precision of your floats and the operations you make on them. See [Python's `decimal.Decimal`](https://docs.python.org/3/library/decimal.html) for documentation on what a decimal data type is. |
| `String`                              | Variable length UTF-8 encoded string data, typically Human-readable.                                                                                                                                                                                                                                                     |
| `Binary`                              | Stores arbitrary, varying length raw binary data.                                                                                                                                                                                                                                                                        |
| `Date`                                | Represents a calendar date.                                                                                                                                                                                                                                                                                              |
| `Time`                                | Represents a time of day.                                                                                                                                                                                                                                                                                                |
| `Datetime`                            | Represents a calendar date and time of day.                                                                                                                                                                                                                                                                              |
| `Duration`                            | Represents a time duration.                                                                                                                                                                                                                                                                                              |
| `Array`                               | Arrays with a known, fixed shape per series; akin to numpy arrays. [Learn more about how arrays and lists differ and how to work with both](../expressions/lists.md).                                                                                                                                                    |
| `List`                                | Homogeneous 1D container with variable length. [Learn more about how arrays and lists differ and how to work with both](../expressions/lists.md).                                                                                                                                                                        |
| `Object`                              | Wraps arbitrary Python objects.                                                                                                                                                                                                                                                                                          |
| `Categorical`                         | Efficient encoding of string data where the categories are inferred at runtime. [Learn more about how categoricals and enums differ and how to work with both](../expressions/categorical-data-and-enums.md).                                                                                                            |
| `Enum`                                | Efficient ordered encoding of a set of predetermined string categories. [Learn more about how categoricals and enums differ and how to work with both](../expressions/categorical-data-and-enums.md).                                                                                                                    |
| `Struct`                              | Composite product type that can store multiple fields. [Learn more about the data type `Struct` in its dedicated documentation section.](../expressions/structs.md).                                                                                                                                                     |
| `Null`                                | Represents null values.                                                                                                                                                                                                                                                                                                  |
