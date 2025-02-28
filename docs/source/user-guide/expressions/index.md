# Expressions

We
[introduced the concept of “expressions” in a previous section](../concepts/expressions-and-contexts.md#expressions).
In this section we will focus on exploring the types of expressions that Polars offers. Each section
gives an overview of what they do and provides additional examples.

<!-- dprint-ignore-start -->
- Essentials:
    - [Basic operations](basic-operations.md) – how to do basic operations on dataframe columns, like arithmetic calculations, comparisons, and other common, general-purpose operations
    - [Expression expansion](expression-expansion.md) – what is expression expansion and how to use it
    - [Casting](casting.md) – how to convert / cast values to different data types
- How to work with specific types of data or data type namespaces:
    - [Strings](strings.md) – how to work with strings and the namespace `str`
    - [Lists and arrays](lists-and-arrays.md) – the differences between the data types `List` and `Array`, when to use them, and how to use them
    - [Categorical data and enums](categorical-data-and-enums.md) – the differences between the data types `Categorical` and `Enum`, when to use them, and how to use them
    - [Structs](structs.md) – when to use the data type `Struct` and how to use it
    - [Missing data](missing-data.md) – how to work with missing data and how to fill missing data
- Types of operations:
    - [Aggregation](aggregation.md) – how to work with aggregating contexts like `group_by`
    - [Window functions](window-functions.md) – how to apply window functions over columns in a dataframe
    - [Folds](folds.md) – how to perform arbitrary computations horizontally across columns
- [User-defined Python functions](user-defined-python-functions.md) – how to apply user-defined Python functions to dataframe columns or to column values
- [Numpy functions](numpy-functions.md) – how to use NumPy native functions on Polars dataframes and series
<!-- dprint-ignore-end -->
