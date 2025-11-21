# Categorical data and enums

A column that holds string values that can only take on one of a limited number of possible values
is a column that holds [categorical data](https://en.wikipedia.org/wiki/Categorical_variable).
Usually, the number of possible values is much smaller than the length of the column. Some typical
examples include your nationality, the operating system of your computer, or the license that your
favorite open source project uses.

When working with categorical data you can use Polars' dedicated types, `Categorical` and `Enum`, to
make your queries more performant. Now, we will see what are the differences between the two data
types `Categorical` and `Enum` and when you should use one data type or the other. We also include
some notes on
[why the data types `Categorical` and `Enum` are more efficient than using the plain string values](#performance-considerations-on-categorical-data-types)
in the end of this user guide section.

## `Enum` vs `Categorical`

In short, you should prefer `Enum` over `Categorical` whenever possible. When the categories are
fixed and known up front, use `Enum`. When you don't know the categories or they are not fixed then
you must use `Categorical`. In case your requirements change along the way you can always cast from
one to the other.

## Data type `Enum`

### Creating an `Enum`

The data type `Enum` is an ordered categorical data type. To use the data type `Enum` you have to
specify the categories in advance to create a new data type that is a variant of an `Enum`. Then,
when creating a new series, a new dataframe, or when casting a string column, you can use that
`Enum` variant.

{{code_block('user-guide/expressions/categoricals', 'enum-example', ['Enum'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:enum-example"
```

### Invalid values

Polars will raise an error if you try to specify a data type `Enum` whose categories do not include
all the values present:

{{code_block('user-guide/expressions/categoricals', 'enum-wrong-value', ['Enum'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:enum-wrong-value"
```

If you are in a position where you cannot know all of the possible values in advance and erroring on
unknown values is semantically wrong, you may need to
[use the data type `Categorical`](#data-type-categorical).

### Category ordering and comparison

The data type `Enum` is ordered and the order is induced by the order in which you specify the
categories. The example below uses log levels as an example of where an ordered `Enum` is useful:

{{code_block('user-guide/expressions/categoricals', 'log-levels', ['Enum'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:log-levels"
```

This example shows that we can compare `Enum` values with a string, but this only works if the
string matches one of the `Enum` values. If we compared the column “level” with any string other
than `"debug"`, `"info"`, `"warning"`, or `"error"`, Polars would raise an exception.

Columns with the data type `Enum` can also be compared with other columns that have the same data
type `Enum` or columns that hold strings, but only if all the strings are valid `Enum` values.

## Data type `Categorical`

The data type `Categorical` can be seen as a more flexible version of `Enum`.

### Creating a `Categorical` series

To use the data type `Categorical`, you can cast a column of strings or specify `Categorical` as the
data type of a series or dataframe column:

{{code_block('user-guide/expressions/categoricals', 'categorical-example', ['Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:categorical-example"
```

Having Polars infer the categories for you may sound strictly better than listing the categories
beforehand, but this inference comes with a performance cost. That is why, whenever possible, you
should use `Enum`. You can learn more by
[reading the subsection about the data type `Categorical` and its encodings](#data-type-categorical-and-encodings).

### Lexical comparison with strings

When comparing a `Categorical` column with a string, Polars will perform a lexical comparison:

{{code_block('user-guide/expressions/categoricals', 'categorical-comparison-string',
['Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:categorical-comparison-string"
```

You can also compare a column of strings with your `Categorical` column, and the comparison will
also be lexical:

{{code_block('user-guide/expressions/categoricals', 'categorical-comparison-string-column',
['Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:categorical-comparison-string-column"
```

Although it is possible to compare a string column with a categorical column, it is typically more
efficient to compare two categorical columns. We will see how to do that next.

### Comparing `Categorical` columns and the string cache

You are told that comparing columns with the data type `Categorical` is more efficient than if one
of them is a string column. So, you change your code so that the second column is also a categorical
column and then you perform your comparison... But Polars raises an exception:

{{code_block('user-guide/expressions/categoricals', 'categorical-comparison-categorical-column',
['Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:categorical-comparison-categorical-column"
```

By default, the values in columns with the data type `Categorical` are
[encoded in the order they are seen in the column](#encodings), and independently from other
columns, which means that Polars cannot compare efficiently two categorical columns that were
created independently.

Enabling the Polars string cache and creating the columns with the cache enabled fixes this issue:

{{code_block('user-guide/expressions/categoricals', 'stringcache-categorical-equality',
['StringCache', 'Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:stringcache-categorical-equality"
```

Note that using [the string cache comes at a performance cost](#using-the-global-string-cache).

### Combining `Categorical` columns

The string cache is also useful in any operation that combines or mixes two columns with the data
type `Categorical` in any way. An example of this is when
[concatenating two dataframes vertically](../getting-started.md#concatenating-dataframes):

{{code_block('user-guide/expressions/categoricals', 'concatenating-categoricals', ['StringCache',
'Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:concatenating-categoricals"
```

In this case, Polars issues a warning complaining about an expensive reenconding that implies taking
a performance hit. Polars then suggests using the data type `Enum` if possible, or using the string
cache. To understand the issue with this operation and why Polars raises an error, please read the
final section about
[the performance considerations of using categorical data types](#performance-considerations-on-categorical-data-types).

### Comparison between `Categorical` columns is lexical

Since Polars 1.32.0, when comparing two columns with data type `Categorical`, Polars always performs
lexical (alphabetical) comparison between the values. The `ordering` parameter has been deprecated
and is now ignored.

Prior to Polars version 1.32.0, when comparing two columns with data type `Categorical`, Polars does
not perform lexical comparison between the values by default. If you want lexical ordering, you need
to specify so when creating the column:

{{code_block('user-guide/expressions/categoricals', 'stringcache-categorical-comparison-lexical',
['StringCache', 'Categorical'])}}

```python exec="on" result="text" session="expressions/categoricals"
--8<-- "python/user-guide/expressions/categoricals.py:stringcache-categorical-comparison-lexical"
```

Otherwise, the order is inferred together with the values:

{{code_block('user-guide/expressions/categoricals', 'stringcache-categorical-comparison-physical',
['StringCache', 'Categorical'])}}

```python
--8<-- "python/user-guide/expressions/categoricals.py:stringcache-categorical-comparison-physical"
```

```
shape: (5,)
Series: '' [bool]
[
	false
	false
	false
	true
	false
]
```

## Performance considerations on categorical data types

This part of the user guide explains

- why categorical data types are more performant than the string literals; and
- why Polars needs a string cache when doing some operations with the data type `Categorical`.

### Encodings

Categorical data represents string data where the values in the column have a finite set of values
(usually way smaller than the length of the column). Storing these values as plain strings is a
waste of memory and performance as we will be repeating the same string over and over again.
Additionally, in operations like joins we have to perform expensive string comparisons.

Categorical data types like `Enum` and `Categorical` let you encode the string values in a cheaper
way, establishing a relationship between a cheap encoding value and the original string literal.

As an example of a sensible encoding, Polars could choose to represent the finite set of categories
as positive integers. With that in mind, the diagram below shows a regular string column and a
possible representation of a Polars column with the categorical data type:

<table>
<tr><th>String Column </th><th>Categorical Column</th></tr>
<tr><td>
<table>
    <thead>
        <tr>
            <th>Series</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
        <tr>
            <td>Polar</td>
        </tr>
    </tbody>
</table>
</td>
<td>
<table>
<tr>
<td>

<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>

</td>
<td>
<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
    </tbody>
</table>
</td>
</tr>
</table>
</td>
</tr>
</table>

The physical `0` in this case encodes (or maps) to the value 'Polar', the value `1` encodes to
'Panda', and the value `2` to 'Brown'. This encoding has the benefit of only storing the string
values once. Additionally, when we perform operations (e.g. sorting, counting) we can work directly
on the physical representation which is much faster than the working with string data.

### Encodings for the data type `Enum` are global

When working with the data type `Enum` we specify the categories in advance. This way, Polars can
ensure different columns and even different datasets have the same encoding and there is no need for
expensive re-encoding or cache lookups.

### Data type `Categorical` and encodings

The fact that the categories for the data type `Categorical` are inferred come at a cost. The main
cost here is that we have no control over our encodings.

Consider the following scenario where we append the following two categorical series:

{{code_block('user-guide/concepts/data-types/categoricals','append',[])}}

Polars encodes the string values in the order they appear. So, the series would look like this:

<table>
<tr><th>cat_series </th><th>cat2_series</th></tr>
<tr><td>
<table>
<tr>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>

</td>
<td>
<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
    </tbody>
</table>

</td>
</tr>
</table>
</td>
<td>
<table>
<tr>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
    </tbody>
</table>

</td>
<td>

<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
        <tr>
            <td>Polar</td>
        </tr>
    </tbody>
</table>

</td>
</tr>
</table>
</td>
</tr>
</table>

Combining the series becomes a non-trivial task which is expensive as the physical value of `0`
represents something different in both series. Polars does support these types of operations for
convenience, however these should be avoided due to its slower performance as it requires making
both encodings compatible first before doing any merge operations.

### Using the global string cache

One way to handle this reencoding problem is to enable the string cache. Under the string cache, the
diagram would instead look like this:

<table>
<tr><th>Series</th><th>String cache</th></tr>
<tr><td>
<table>
<tr><th>cat_series</th><th>cat2_series</th></tr>
<tr>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>
</td>
<td>
<table>
    <thead>
        <tr>
            <th>Physical</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>1</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>2</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
        <tr>
            <td>0</td>
        </tr>
    </tbody>
</table>
</td>
</tr>
</table>
</td>
<td>
<table>
    <thead>
        <tr>
            <th>Categories</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Polar</td>
        </tr>
        <tr>
            <td>Panda</td>
        </tr>
        <tr>
            <td>Brown</td>
        </tr>
    </tbody>
</table>
</td>
</tr>
</table>

When you enable the string cache, strings are no longer encoded in the order they appear on a
per-column basis. Instead, the encoding is shared across columns. The value 'Polar' will always be
encoded by the same value for all categorical columns created under the string cache. Merge
operations (e.g. appends, joins) become cheap again as there is no need to make the encodings
compatible first, solving the problem we had above.

However, the string cache does come at a small performance hit during construction of the series as
we need to look up or insert the string values in the cache. Therefore, it is preferred to use the
data type `Enum` if you know your categories in advance.
