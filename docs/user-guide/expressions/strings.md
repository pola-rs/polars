# Strings

The following section discusses operations performed on `Utf8` strings, which are a frequently used `DataType` when working with `DataFrames`. However, processing strings can often be inefficient due to their unpredictable memory size, causing the CPU to access many random memory locations. To address this issue, Polars utilizes `Arrow` as its backend, which stores all strings in a contiguous block of memory. As a result, string traversal is cache-optimal and predictable for the CPU.

String processing functions are available in the `str` namespace.

##### Accessing the string namespace

The `str` namespace can be accessed through the `.str` attribute of a column with `Utf8` data type. In the following example, we create a column named `animal` and compute the length of each element in the column in terms of the number of bytes and the number of characters. If you are working with ASCII text, then the results of these two computations will be the same, and using `lengths` is recommended since it is faster.

{{code_block('user-guide/expressions/strings','df',['str.len_bytes','str.len_chars'])}}

```python exec="on" result="text" session="user-guide/strings"
--8<-- "python/user-guide/expressions/strings.py:setup"
--8<-- "python/user-guide/expressions/strings.py:df"
```

#### String parsing

`Polars` offers multiple methods for checking and parsing elements of a string. Firstly, we can use the `contains` method to check whether a given pattern exists within a substring. Subsequently, we can extract these patterns and replace them using other methods, which will be demonstrated in upcoming examples.

##### Check for existence of a pattern

To check for the presence of a pattern within a string, we can use the contains method. The `contains` method accepts either a regular substring or a regex pattern, depending on the value of the `literal` parameter. If the pattern we're searching for is a simple substring located either at the beginning or end of the string, we can alternatively use the `starts_with` and `ends_with` functions.

{{code_block('user-guide/expressions/strings','existence',['str.contains', 'str.starts_with','str.ends_with'])}}

```python exec="on" result="text" session="user-guide/strings"
--8<-- "python/user-guide/expressions/strings.py:existence"
```

##### Extract a pattern

The `extract` method allows us to extract a pattern from a specified string. This method takes a regex pattern containing one or more capture groups, which are defined by parentheses `()` in the pattern. The group index indicates which capture group to output.

{{code_block('user-guide/expressions/strings','extract',['str.extract'])}}

```python exec="on" result="text" session="user-guide/strings"
--8<-- "python/user-guide/expressions/strings.py:extract"
```

To extract all occurrences of a pattern within a string, we can use the `extract_all` method. In the example below, we extract all numbers from a string using the regex pattern `(\d+)`, which matches one or more digits. The resulting output of the `extract_all` method is a list containing all instances of the matched pattern within the string.

{{code_block('user-guide/expressions/strings','extract_all',['str.extract_all'])}}

```python exec="on" result="text" session="user-guide/strings"
--8<-- "python/user-guide/expressions/strings.py:extract_all"
```

##### Replace a pattern

We have discussed two methods for pattern matching and extraction thus far, and now we will explore how to replace a pattern within a string. Similar to `extract` and `extract_all`, Polars provides the `replace` and `replace_all` methods for this purpose. In the example below we replace one match of `abc` at the end of a word (`\b`) by `ABC` and we replace all occurrence of `a` with `-`.

{{code_block('user-guide/expressions/strings','replace',['str.replace','str.replace_all'])}}

```python exec="on" result="text" session="user-guide/strings"
--8<-- "python/user-guide/expressions/strings.py:replace"
```

#### API documentation

In addition to the examples covered above, Polars offers various other string manipulation methods for tasks such as formatting, stripping, splitting, and more. To explore these additional methods, you can go to the API documentation of your chosen programming language for Polars.
