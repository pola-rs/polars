# Strings

The following section discusses operations performed on string data, which is a frequently used data
type when working with dataframes. String processing functions are available in the namespace `str`.

Working with strings in other dataframe libraries can be highly inefficient due to the fact that
strings have unpredictable lengths. Polars mitigates these inefficiencies by
[following the Arrow Columnar Format specification](../concepts/data-types-and-structures.md#data-types-internals),
so you can write performant data queries on string data too.

## The string namespace

When working with string data you will likely need to access the namespace `str`, which aggregates
40+ functions that let you work with strings. As an example of how to access functions from within
that namespace, the snippet below shows how to compute the length of the strings in a column in
terms of the number of bytes and the number of characters:

{{code_block('user-guide/expressions/strings','df',['str.len_bytes','str.len_chars'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:df"
```

!!! note

    If you are working exclusively with ASCII text, then the results of the two computations will be the same and using `len_bytes` is recommended since it is faster.

## Parsing strings

Polars offers multiple methods for checking and parsing elements of a string column, namely checking
for the existence of given substrings or patterns, and counting, extracting, or replacing, them. We
will demonstrate some of these operations in the upcoming examples.

### Check for the existence of a pattern

We can use the function `contains` to check for the presence of a pattern within a string. By
default, the argument to the function `contains` is interpreted as a regular expression. If you want
to specify a literal substring, set the parameter `literal` to `True`.

For the special cases where you want to check if the strings start or end with a fixed substring,
you can use the functions `starts_with` or `ends_with`, respectively.

{{code_block('user-guide/expressions/strings','existence',['str.contains',
'str.starts_with','str.ends_with'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:existence"
```

### Regex specification

Polars relies on the Rust crate `regex` to work with regular expressions, so you may need to
[refer to the syntax documentation](https://docs.rs/regex/latest/regex/#syntax) to see what features
and flags are supported. In particular, note that the flavor of regex supported by Polars is
different from Python's module `re`.

### Extract a pattern

The function `extract` allows us to extract patterns from the string values in a column. The
function `extract` accepts a regex pattern with one or more capture groups and extracts the capture
group specified as the second argument.

{{code_block('user-guide/expressions/strings','extract',['str.extract'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:extract"
```

To extract all occurrences of a pattern within a string, we can use the function `extract_all`. In
the example below, we extract all numbers from a string using the regex pattern `(\d+)`, which
matches one or more digits. The resulting output of the function `extract_all` is a list containing
all instances of the matched pattern within the string.

{{code_block('user-guide/expressions/strings','extract_all',['str.extract_all'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:extract_all"
```

### Replace a pattern

Akin to the functions `extract` and `extract_all`, Polars provides the functions `replace` and
`replace_all`. These accept a regex pattern or a literal substring (if the parameter `literal` is
set to `True`) and perform the replacements specified. The function `replace` will make at most one
replacement whereas the function `replace_all` will make all the non-overlapping replacements it
finds.

{{code_block('user-guide/expressions/strings','replace',['str.replace', 'str.replace_all'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:replace"
```

## Modifying strings

### Case conversion

Converting the casing of a string is a common operation and Polars supports it out of the box with
the functions `to_lowercase`, `to_titlecase`, and `to_uppercase`:

{{code_block('user-guide/expressions/strings','casing', ['str.to_lowercase', 'str.to_titlecase',
'str.to_uppercase'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:casing"
```

### Stripping characters from the ends

Polars provides five functions in the namespace `str` that let you strip characters from the ends of
the string:

| Function            | Behaviour                                                             |
| ------------------- | --------------------------------------------------------------------- |
| `strip_chars`       | Removes leading and trailing occurrences of the characters specified. |
| `strip_chars_end`   | Removes trailing occurrences of the characters specified.             |
| `strip_chars_start` | Removes leading occurrences of the characters specified.              |
| `strip_prefix`      | Removes an exact substring prefix if present.                         |
| `strip_suffix`      | Removes an exact substring suffix if present.                         |

??? info "Similarity to Python string methods"

    `strip_chars` is similar to Python's string method `strip` and `strip_prefix`/`strip_suffix`
    are similar to Python's string methods `removeprefix` and `removesuffix`, respectively.

It is important to understand that the first three functions interpret their string argument as a
set of characters whereas the functions `strip_prefix` and `strip_suffix` do interpret their string
argument as a literal string.

{{code_block('user-guide/expressions/strings', 'strip', ['str.strip_chars', 'str.strip_chars_end',
'str.strip_chars_start', 'str.strip_prefix', 'str.strip_suffix'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:strip"
```

If no argument is provided, the three functions `strip_chars`, `strip_chars_end`, and
`strip_chars_start`, remove whitespace by default.

### Slicing

Besides [extracting substrings as specified by patterns](#extract-a-pattern), you can also slice
strings at specified offsets to produce substrings. The general-purpose function for slicing is
`slice` and it takes the starting offset and the optional _length_ of the slice. If the length of
the slice is not specified or if it's past the end of the string, Polars slices the string all the
way to the end.

The functions `head` and `tail` are specialised versions used for slicing the beginning and end of a
string, respectively.

{{code_block('user-guide/expressions/strings', 'slice', [], ['str.slice', 'str.head', 'str.tail'],
['str.str_slice', 'str.str_head', 'str.str_tail'])}}

```python exec="on" result="text" session="expressions/strings"
--8<-- "python/user-guide/expressions/strings.py:slice"
```

## API documentation

In addition to the examples covered above, Polars offers various other string manipulation
functions. To explore these additional methods, you can go to the API documentation of your chosen
programming language for Polars.
